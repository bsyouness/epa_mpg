import xlwings as xw
import pandas as pd
import re
import operator
from datetime import datetime
from math import isnan
import numpy as np

######################################################################################################################
# Load databases.
######################################################################################################################
epa_original = pd.read_csv(r'X:\EPA_MPG\epa_data.csv', encoding='utf8', dtype=unicode)
vin_original = pd.read_csv(r'X:\EPA_MPG\vin_with_vtyp3.csv', encoding='utf8', dtype=unicode)

# Get rid of the 'Results_0_' string in the column titles
# pattern = re.compile(u'Results_0_(.+)')
# vin_original.columns = [pattern.search(x).groups()[0] 
# 	if pattern.search(x) else x for x in vin_original.columns]

# Define integer id based on error code from VIN database. 
vin_original['error_id'] = vin_original.ErrorCode.apply(
	lambda x: re.match('([0-9]+).*', x).groups()[0])

# Define columns on which the merge will be performed.
epa_cols = [
	'make',
	'model',
	'year',
	'fuelType1',
	'drive',
	'transmission_type',
	'transmission_speeds',
	'cylinders',
	'displ',
	]
vin_cols = [
	'Make',
	'Model',
	'ModelYear',
	'FuelTypePrimary',
	'DriveType',
	'TransmissionStyle', 
	'TransmissionSpeeds',
	'EngineCylinders',
	'DisplacementL',
	]

# Get rid of undesirable columns.
""" Previously vin_keep_cols included:
	'ModelYear', 'Series', 'FuelTypePrimary', 'Trim2', 'Doors', 'Make', 'Series2', 
	'BodyClass', 'DisplacementL', 'EngineCylinders', 'Trim', 'Model', 'FuelTypeSecondary', 
	'ErrorCode', 'VehicleType', 'Manufacturer', 'DriveType', 'TransmissionStyle', 'TransmissionSpeeds'
"""
vin_keep_cols = ['VIN', 'VehicleType', 'BodyClass', 'error_id', 'Series', 'vtyp3'] + vin_cols 
vin_original.drop([x for x in vin_original.columns if x not in vin_keep_cols], axis=1, inplace=True)
epa_keep_cols = ['trany', 'city08', 'city08U', 'comb08', 'comb08U', 'highway08', 'highway08U', 'VClass',] + epa_cols
epa_original.drop([x for x in epa_original.columns if x not in epa_keep_cols], axis=1, inplace=True)

# Rename the VIN dataframe columns to be the same as the EPA dataframe columns.
vin_original = vin_original.rename(columns=dict(zip(vin_cols, epa_cols)))

# Get rid of rows where certain info is missing.
essential_cols = 'make, model, year, fuelType1'.split(', ')
for col in essential_cols:
	vin_original = vin_original.loc[~vin_original[col].isnull()]

# Replace missing values (nan) with u'-1'.
vin_original, epa_original = [
	df.apply(lambda x: pd.Series.fillna(x, u'-1')) for df in (vin_original, epa_original)]

# Make everything lower case and trim white spaces.
vin_original, epa_original = [
	df.applymap(lambda s: s.lower().strip()) for df in (vin_original, epa_original)]

# Get rid of undesirable vehicles.
filter_out_strs = ['incomplete vehicle', 'trailer', 'motorcycle', 'bus', 'low speed vehicle (lsv)']
vin_original = vin_original.loc[~vin_original['VehicleType'].isin(filter_out_strs)]
vin_original = vin_original.loc[~vin_original['BodyClass'].str.contains('incomplete')]

# Get rid of tranmission type in model for epa_original data. 
pattern = r'(.+)\s[24a]wd.*'
epa_original['model'] = epa_original['model'].apply(
	lambda x: re.search(pattern, x).groups()[0] if re.search(pattern, x) else x)

# Get rid of duplicates in fields (e.g. 'gasoline, gasoline', or 'audi, audi').
def del_duplicate_in_str(s):
	def _del_duplicate_in_str(s):
		found = pattern.search(s)
		if found:
			s0, s1 = found.groups()
			if pattern.search(s0):
				return _del_duplicate_in_str(s0)
			elif s0 == s1:
				return s0
		return s

	if not isinstance(s, unicode):
		return s
	pattern = re.compile('(.*), (.*)')
	return _del_duplicate_in_str(s)	
vin_original = vin_original.applymap(del_duplicate_in_str)

# Modify makes. 
drop_makes = \
	'''volvo truck
	western star
	whitegmc
	winnebago
	winnebago industries, inc.
	workhorse
	ai-springfield
	autocar industries
	capacity of texas
	caterpillar
	e-one
	freightliner
	kenworth
	mack
	navistar
	peterbilt
	pierce manufacturing
	spartan motors chassis
	terex advance mixer
	the vehicle production group
	utilimaster motor corporation
	international
	'''.split('\n\t')
vin_original = vin_original.loc[~vin_original.make.isin(drop_makes)]

mapping = {
	'vin': {
		'fuelType1':	{
			'compressed natural gas (cng)':									'natural gas',
			'liquefied petroleum gas (propane or lpg)':						'natural gas',
			'liquefied natural gas (lng)':									'natural gas',
			'gasoline, diesel':												'gasoline',
			'diesel, gasoline':												'gasoline',
			'ethanol (e85)':												'ethanol',
			'compressed natural gas (cng), gasoline':						'gasoline',
			'gasoline, compressed natural gas (cng)':						'gasoline',
			'compressed hydrogen / hydrogen':								'hydrogen',
			'fuel cell':													'hydrogen',
			},
		'drive':	{
			'4x2':																'two',
			'6x6':																'all',
			'6x2':																'two',
			'8x2':																'two',
			'rwd/ rear wheel drive':											'two',
			'fwd/front wheel drive':											'two',
			'4x2, rwd/ rear wheel drive':										'two',
			'4x2, fwd/front wheel drive':										'two',
			'rwd/ rear wheel drive, 4x2':										'two',
			'fwd/front wheel drive, 4x2':										'two',
			'4wd/4-wheel drive/4x4':											'all',
			'awd/all wheel drive':												'all',
			},
		'transmission_type':	{
			'manual/standard': 												'manu',
			'automated manual transmission (amt)': 							'manu',
			'manual/standard, manual/standard': 							'manu',
			'dual-clutch transmission (dct)': 								'manu',
			'continuously variable transmission (cvt)': 					'auto',
			'automatic': 													'auto',
			'automatic, continuously variable transmission (cvt)': 			'auto',
			}
		},
	'epa': {
		'fuelType1':	{
			'regular gasoline':			'gasoline',
			'premium gasoline':			'gasoline',
			'midgrade gasoline':		'gasoline',
			},
		'drive':	{
			'rear-wheel drive':				'two',
			'front-wheel drive':			'two',
			'2-wheel drive':				'two',
			'all-wheel drive':				'all',
			'4-wheel drive':				'all',
			'4-wheel or all-wheel drive':	'all',
			'part-time 4-wheel drive':		'all',
			},
		}
	}
## Modify fuel type, drive type for epa and vin, and transmission type for vin. 
for (df, df_name) in ((epa_original, 'epa'), (vin_original, 'vin')):
	for item in mapping[df_name]:
		df[item + '_mod'] = df[item].replace(mapping[df_name][item])

# Make years ints.
epa_original.year = epa_original.year.apply(int)
vin_original.year = vin_original.year.apply(int)

# Take out flexible fuel vehicles. 
vin_original, epa_original = [df.loc[~df.fuelType1.str.contains('flexible')] 
	for df in (vin_original, epa_original)]
def mod_electric_vehicles(df):
	df.loc[df.model.str.contains('plug'), 'fuelType1'] = 'phev'
	df.loc[(df.model.str.contains('(hev|hybrid)')) | ((df.fuelType1.str.contains('gasoline')) &
			(df.fuelType1.str.contains('electric'))), 'fuelType1'] = 'hev'
	df.loc[(df.model.str.contains('bev')) | (df.model.str.contains('electric')) | 
		(df.fuelType1.str.contains('electric')), 'fuelType1'] = 'bev'
	return df
epa_original = mod_electric_vehicles(epa_original)
vin_original = mod_electric_vehicles(vin_original)

# Add IDs
vin_original['VIN_ID'] = range(1, len(vin_original) + 1)
epa_original['EPA_ID'] = range(1, len(epa_original) + 1)

# Add model_mod.
vin_original['model_mod'] = vin_original['model']
epa_original['model_mod'] = epa_original['model']

# Turn pick-up into pickup. 
vin_original['model_mod'] = vin_original.model_mod.str.replace('pick-up', 'pickup')

# Turn anything that looks like this: texttext-123123 (\w+-\d+) into texttext123123 (drop the dash)
# e.g. f-350
vin_original.loc[vin_original.model.str.contains(r'[^\d]+-\d+.*'), 'model_mod'] = \
	vin_original.loc[vin_original.model.str.contains(r'[^\d]+-\d+.*'), 'model_mod'].str.replace('-', '')
epa_original.loc[epa_original.model.str.contains(r'[^\d]+-\d+.*'), 'model_mod'] = \
	epa_original.loc[epa_original.model.str.contains(r'[^\d]+-\d+.*'), 'model_mod'].str.replace('-', '')

# Split rows that contain '/' into several rows. 
vin_expanded = pd.concat(
		[pd.Series(np.append(row[[col for col in vin_original.columns if col != 'model_mod']].values, [x]))
			for _, row in vin_original.iterrows() 
			for x in map(unicode.strip, re.findall(r'[\w ]+', row['model_mod']))],
		axis=1).transpose()
def split_row(s, separator):
	pattern1 = re.compile(r'(.*?)(?=\S*[{}] *\S*?)(\S*)(.*)'.format(separator))
	if pattern1.match(s):
		groups = pattern1.match(s).groups()
		parts = map(unicode.strip, re.split(separator, groups[1]))
		pattern2 = re.compile(r'([\w\W]*?)(\d+)$')
		subparts = []
		for part in parts:
			if re.match(pattern2, part):
				subparts.append(re.match(pattern2, part).groups())
		if len(subparts) == len(parts):
			zip_subparts =  zip(*subparts)
			def find_non_blank(t):
				try: return reduce(lambda x, xs: x if x != '' else xs[0], t)
				except: return ''
			default = map(find_non_blank, zip_subparts)
			parts = [(subpart1 or default[0]) + (subpart2 or default[1]) for (subpart1, subpart2) in subparts]
		return map(''.join, [[groups[0], x, groups[2]] for x in parts])
	else:
		return [s]
epa_expanded = pd.concat(
		[pd.Series(np.append(row[[col for col in epa_original.columns if col != 'model_mod']].values, [x]))
			for _, row in epa_original.iterrows() 
			for x in split_row(row['model_mod'], '\/|,')],
		axis=1).transpose()
vin_expanded.columns, epa_expanded.columns = vin_original.columns, epa_original.columns
vin_original, epa_original = vin_expanded, epa_expanded

# Reset index.
vin_original = vin_original.reset_index(drop=True)
epa_original = epa_original.reset_index(drop=True)

######################################################################################################################
# Modify the basics.
######################################################################################################################
# Create a copy of the VIN and EPA databases.
epa_1 = epa_original.copy()
vin_1 = vin_original.copy()

# Merge the vin_1 db with the ERG db.
erg = pd.read_csv('X:\EPA_MPG\ERG_output.csv', header=None, dtype=unicode)
erg.columns = ['VIN', 'counts']
erg = erg[erg['counts'] != '.']
erg['counts'] = erg['counts'].astype(int)
erg = erg.loc[erg['counts'] > 10]
erg['VIN'] = erg['VIN'].apply(str.lower)
vin_1 = pd.merge(vin_1, erg, how='inner')

# Modify transmission information
## In vin_1 DB: turn transmission speeds into integers then strings.
def try_int_unicode(a):
	try:
		return unicode(int(a))
	except:
		return unicode(a)
vin_1['transmission_speeds_mod'] = vin_1['transmission_speeds'].apply(try_int_unicode)
## In epa_1 DB: transform info in epa_1 database to get trammission speeds and types.
## Transmission speeds.
def get_transmission_speeds(s):
	try:
		return re.search(r'\d+', s).group()
	except:
		return None
## Transmission type.
def get_transmission_type(s):
	if isinstance(s, unicode):
		if re.search(r'auto', s):
			return "auto"
		else:
			return "manu"
## Apply to epa_1.
epa_1['transmission_speeds_mod'] = epa_1['transmission_speeds'] = epa_1.trany.apply(get_transmission_speeds)
epa_1['transmission_type_mod'] = epa_1['transmission_type'] = epa_1.trany.apply(get_transmission_type)

# Round displacement in both databases.
def convert_displacement(s):
	try:
		return round(float(s), 1)
	except:
		return None
for df in (epa_1, vin_1):
	df['displ_mod'] = df['displ'].apply(convert_displacement)

# Update -1 to default values.
for df in (epa_1, vin_1):
	df['fuelType1_mod'] = df['fuelType1_mod'].replace({'-1': 'gasoline'}) 
	df['drive_mod'] = df['drive_mod'].replace({'-1': 'two'}) 
epa_1['cylinders'] = epa_1['cylinders'].fillna(-1).astype(int).astype(unicode)
vin_1['cylinders'] = vin_1['cylinders'].astype(unicode)

# Change type of mpg values to be floats. 
mpg_list = 'highway08, highway08U, comb08, comb08U, city08, city08U'.split(', ')
epa_1[mpg_list] = epa_1[mpg_list].astype(float)

# Reset index.
vin_1 = vin_1.reset_index(drop=True)
epa_1 = epa_1.reset_index(drop=True)

######################################################################################################################
# Modify makes, models, etc.
######################################################################################################################
epa = epa_1.copy()
vin = vin_1.copy()
# Modify makes. 
## Ram.
vin.ix[(vin.make == 'ram'), 'make'] = 'dodge'
epa.ix[(epa.make == 'ram'), 'make'] = 'dodge'
## Ford. 
epa.ix[(epa.make == 'ford') & (epa.model == 'escort zx2'), 'model_mod'] = 'zx2'
## Chevrolet.
vin.ix[(vin.make == 'geo'), 'make'] = 'chevrolet'
epa.ix[(epa.make == 'geo'), 'make'] = 'chevrolet'
epa.ix[epa.make == 'gmc', 'make'] = 'chevrolet'
vin.ix[vin.make == 'gmc', 'make'] = 'chevrolet'
## Toyota. 
vin.ix[vin.model.str.contains('scion'), 'make'] = 'scion'
## Chrysler.
### Replace Chrysler with Dodge for model Caravan in VIN.
vin.ix[(vin.make == 'chrysler') & (vin.model_mod == 'caravan'), 'make'] = 'dodge'
## Sprinter
vin.ix[(vin.make == 'sprinter (dodge or freightliner)'), 'make'] = 'dodge'

# Modify models. 
## All models with displacements in the model name, e.g. '190e 2.3-16'
def mod_models_w_displ(s):
	pattern = re.compile(r'(.*)\d\.\d(.*)')
	groups = re.match(pattern, s).groups()
	return groups[0].strip() or groups[1].strip()
## Lincoln.
### Replace 'zephyr' with 'mkz' for VIN for make 'lincoln'
vin.loc[(vin.make == 'lincoln') & (vin.model.str.contains('zephyr')), 'model_mod'] = 'mkz'
## Ford.
### Replace 'ltd crown victoria' with 'crown victoria' in EPA for make 'ford'
epa.loc[(epa.make == 'ford') & (epa.model.str.contains('ltd crown victoria')), 'model_mod'] = 'crown victoria'
### Replace 'crown victoria' with 'crown victoria police' in VIN for make 'ford' when `series == 'police interceptor'`
vin.loc[(vin.make == 'ford') & (vin.model.str.contains('crown victoria')) & (vin.Series == 'police interceptor'),
	'model_mod'] = 'crown victoria police'
## Chrysler.
### Replace '300c' and '300c/srt' with '300' in EPA
epa.loc[(epa.make == 'chrysler') & (epa.model.str.contains('300')), 'model'] = '300'
## Saturn.
### Replace L100/200 with LS1 in EPA when year is larger than 2001 (inclusive)
epa.loc[(epa.make == 'saturn') & (epa.model == 'l100/200') & (epa.year >= 2001), 'model_mod'] = 'ls1'
vin.loc[(vin.make == 'saturn') & (vin.model_mod != 'ls1'), 'model_mod'] = \
	vin.loc[(vin.make == 'saturn') & (vin.model_mod != 'ls1'), 'model_mod'].apply(
		lambda s: re.match(r'([^\d]+)[\d]', s).groups()[0] if re.match(r'([^\d]+)[\d]', s) else s)
## Mercedes. 
vin_mercedes_index = vin.loc[(vin.make == 'mercedes-benz') & (vin.Series != '-1'), 'Series'].index
vin.ix[vin_mercedes_index, 'model_mod'] = vin.ix[vin_mercedes_index, 'Series']
vin.ix[vin_mercedes_index, 'model_mod'] = \
	vin.ix[vin_mercedes_index, 'model_mod'].str.replace('amg', '').str.strip()
epa.ix[(epa.make == 'mercedes-benz'), 'model_mod'] = \
	epa.ix[(epa.make == 'mercedes-benz'), 'model_mod'].str.replace('amg', '').str.strip()
def mod_mercedes(s):
	pattern = re.compile(r'.*?(\w*\d+\w*).*')
	match = re.match(pattern, s)
	if match:
		return match.groups()[0]
	else:
		return re.findall(r'\w+', s)[0]
vin.ix[vin_mercedes_index, 'model_mod'] = vin.ix[vin_mercedes_index, 'model_mod'].apply(mod_mercedes)
epa.ix[(epa.make == 'mercedes-benz'), 'model_mod'] = \
	epa.ix[(epa.make == 'mercedes-benz'), 'model_mod'].apply(mod_mercedes)
## Toyota. 
vin.ix[vin.make == 'scion', ['model', 'model_mod']] = \
	vin.ix[vin.make == 'scion', 'model_mod'].apply(lambda x: x.split(' ')[1] if len(x.split(' '))>1 else x)
epa.ix[epa.model == 'camry solara', 'model_mod'] = 'solara'
vin.ix[vin.model.str.contains('4-runner'), 'model_mod'] = '4runner'
## Mazda.
def delete_mazda(s):
	match = re.match('mazda(.*)', s)
	if match:
		return match.groups()[0]
	else:
		return s
vin.ix[vin['make'] == 'mazda', 'model_mod'] = \
	vin.ix[vin['make'] == 'mazda', 'model_mod'].apply(delete_mazda)
epa.ix[(epa.make == 'mazda') & (epa.model_mod.str.contains('^b\d')), 'model_mod'] = \
	epa.ix[(epa.make == 'mazda') & (epa.model_mod.str.contains('^b\d')), 'model_mod'].apply(lambda x: x[0])
## John Cooper Works. 
pattern = re.compile('john cooper works(.*)')
epa['model_mod'] = epa['model_mod'].apply(
	lambda x: 'jcw'+pattern.match(x).groups()[0] if(pattern.match(x)) else x)
## Chevrolet.
## Get rid of anything like s10 from the name of the model. 
# epa.ix[(epa.make == 'chevrolet') & (epa.model_mod.str.contains(r'.*?(\s|^)([^\d]+)(\s|$)')), 'model_mod'] = \
# 	epa.ix[(epa.make == 'chevrolet') & (epa.model_mod.str.contains(r'.*?(\s|^)([^\d]+)(\s|$)')), 'model_mod'].apply(
# 		lambda x: re.match(r'.*?(\s|^)([^\d]+)(\s|$)', x).groups()[1])
### s10 models. 
epa.ix[(epa.make == 'chevrolet') & (epa.model.str.contains('(^|\s)blazer($|\s)')), 'model_mod'] = 'blazer'
vin.ix[(vin.make == 'chevrolet') & (vin.model.str.contains('(^|\s)blazer($|\s)')), 'model_mod'] = 'blazer'
epa.ix[(epa.make == 'chevrolet') & (epa.model.str.contains('suburban')), 'model_mod'] = 'suburban'
vin.ix[(vin.make == 'chevrolet') & (vin.model.str.contains('suburban')), 'model_mod'] = 'suburban'
epa.ix[(epa.make == 'chevrolet') & (epa.model.str.contains('s10|s-10')), 'model_mod'] = 's'
vin.ix[(vin.make == 'chevrolet') & (vin.model.str.contains('s10|s-10')), 'model_mod'] = 's'
### Geo Metro model. 
vin.ix[(vin.make == 'chevrolet') & (vin.model.str.contains('geo metro')), 'model_mod'] = 'metro'
### Replace gmt-400 with c when drive_mod = 2 and with k otherwise. 
vin.ix[(vin.make == 'chevrolet') & (vin.model == 'gmt-400') & (vin.drive_mod == 'two'), 'model_mod'] = 'c'
vin.ix[(vin.make == 'chevrolet') & (vin.model == 'gmt-400') & (vin.drive_mod == 'all'), 'model_mod'] = 'k'
### Replace '^\D\d' with the first letter. 
pattern = r'^(\D+)\s*\d+'
ind = epa.ix[(epa.make.isin(['chevrolet', 'dodge'])) & (epa.model_mod.str.contains(pattern)), 'model_mod'].index
epa.ix[ind, 'model_mod'] = epa.ix[ind, 'model_mod'].str.extract(pattern)
ind = vin.ix[(vin.make.isin(['chevrolet', 'dodge'])) & (vin.model_mod.str.contains(pattern)), 'model_mod'].index
vin.ix[ind, 'model_mod'] = vin.ix[ind, 'model_mod'].str.extract(pattern)
## Keep only first word of the model. 
epa['model_mod'] = epa['model_mod'].apply(lambda x: x.strip().split(' ')[0].split('-')[0])
vin['model_mod'] = vin['model_mod'].apply(lambda x: x.strip().split(' ')[0].split('-')[0])

# Reset index.
vin = vin.reset_index(drop=True)
epa = epa.reset_index(drop=True)

######################################################################################################################
# Merge databases. 
######################################################################################################################
def merge():
	# Perform the merge.
	## Merge. 
	mod_cols = [
		'model',
		'fuelType1',
		'drive',
		'transmission_speeds',
		'transmission_type',
		'displ',
		]
	append_to_list = lambda l, s: map(''.join, zip(l, [s] * len(l)))
	mod_cols_mod = append_to_list(mod_cols, '_mod')

	# # Clean up columns. 
	# vin.drop([x for x in vin.columns if x not in mod_cols_mod + epa_cols + ['VIN_ID', 'VIN', 'counts']], 
	# 	axis=1, inplace=True)

	# on_cols = [x for x in epa_cols + mod_cols_mod if x not in mod_cols]
	on_cols = [
		'make',
		'model_mod',
		'year',
		'fuelType1_mod',
		'drive_mod',
		'displ_mod',
		'cylinders',
		'transmission_speeds_mod',
		'transmission_type_mod',
		'',
		]

	matched_vins = pd.DataFrame(columns=list(vin.columns) + ['matched_on'])

	for _ in range(7):
		on_cols = on_cols[:-1]
		remaining_vins = vin.loc[~vin.VIN_ID.isin(matched_vins.VIN_ID)]
		inner_join = pd.merge(remaining_vins.reset_index(), epa, how='inner', on=on_cols).set_index('index')
		inner_join['matched_on'] = len(on_cols)
		matched_vins = pd.concat([inner_join, matched_vins])

	return matched_vins

# Merge VIN and EPA db. 
matched_vins = merge()

matched_vins_ids = matched_vins[['VIN_ID', 'EPA_ID', 'matched_on']]

vin_vin = vin.rename(columns = dict(zip(vin.columns, vin.columns + '_vin')))
epa_epa = epa.rename(columns = dict(zip(epa.columns, epa.columns + '_epa')))
# Matched VINs.
matched_vins_simple_id = pd.merge(pd.merge(matched_vins_ids, vin_vin, left_on='VIN_ID', right_on='VIN_ID_vin'), 
	epa_epa, left_on='EPA_ID', right_on='EPA_ID_epa')
matched_vins_simple_id.drop(['EPA_ID', 'VIN_ID'], axis=1, inplace=True)
matched_vins_no_dupes = matched_vins_simple_id.sort_values('EPA_ID_epa', ascending=False).drop_duplicates(subset='VIN_vin')
vins_no_dupes = vin_vin.drop_duplicates(subset='VIN_vin')
print('Merge fraction weighted: {:.2%}'.format(float(matched_vins_no_dupes['counts_vin'].sum())/vins_no_dupes['counts_vin'].sum()))

######################################################################################################################
# Generate output files. 
######################################################################################################################
vins_matched = matched_vins_simple_id.VIN_ID_vin
epas_matched = matched_vins_simple_id.EPA_ID_epa
# Not matched vins.
not_matched_vins = vin_vin.loc[~vin_vin.VIN_ID_vin.isin(vins_matched)]
not_matched_epas = epa_epa.loc[~epa_epa.EPA_ID_epa.isin(epas_matched)]
not_matched = pd.concat([not_matched_epas, not_matched_vins])
# Both.
all_records = pd.concat([not_matched, matched_vins_simple_id])
all_records_no_dupes = pd.concat([not_matched, matched_vins_no_dupes])
ordered_cols = [
	'matched_on',
	'EPA_ID_epa',
	'make_epa',
	'model_epa',
	'model_mod_epa',
	'year_epa',
	'fuelType1_epa',
	'fuelType1_mod_epa',
	'drive_epa',
	'drive_mod_epa',
	'displ_epa',
	'displ_mod_epa',
	'cylinders_epa',
	'transmission_speeds_epa',
	'transmission_speeds_mod_epa',
	'transmission_type_epa',
	'transmission_type_mod_epa',
	'comb08_epa',
	'comb08_epa',
	'highway08_epa',
	'highway08_epa',
	'city08_epa',
	'city08_epa',
	'VClass_epa',
	'trany_epa',
	'VIN_ID_vin',
	'VIN_vin',
	'counts_vin',
	'make_vin',
	'model_vin',
	'model_mod_vin',
	'year_vin',
	'fuelType1_vin',
	'fuelType1_mod_vin',
	'drive_vin',
	'drive_mod_vin',
	'displ_mod_vin',
	'displ_vin',
	'cylinders_vin',
	'transmission_speeds_vin',
	'transmission_speeds_mod_vin',
	'transmission_type_vin',
	'transmission_type_mod_vin',
	'Series_vin',
	'VehicleType_vin',
	'BodyClass_vin',
	'error_id_vin',
	]
# all_records[ordered_cols].to_csv('all_records.csv', encoding='utf8')
# all_records_no_dupes[ordered_cols].to_csv('all_records_no_dupes.csv', encoding='utf8')
# matched_vins_simple_id.to_csv('matched_vins.csv', encoding = 'utf8')

# For debugging.
matched_vins_simple = pd.merge(pd.merge(matched_vins_ids, vin, on='VIN_ID'), epa, on='EPA_ID')
vins_matched_for_comparison = matched_vins_simple.VIN_ID
epas_matched_for_comparison = matched_vins_simple.EPA_ID
not_matched_vins_for_comparison = vin.loc[~vin.VIN_ID.isin(vins_matched_for_comparison)]
not_matched_epas_for_comparison = epa.loc[~epa.EPA_ID.isin(epas_matched_for_comparison)]
not_matched_for_comparison = pd.concat([not_matched_epas_for_comparison, 
	not_matched_vins_for_comparison])

not_matched_out = not_matched_for_comparison[[
	'make',
	'model_mod',
	'model',
	'year',
	'fuelType1_mod',
	'drive_mod',
	'displ_mod',
	'cylinders',
	'transmission_speeds_mod',
	'transmission_type_mod',
	'EPA_ID',
	'VIN_ID',
	'VIN',
	'counts',
	'BodyClass',
	'VehicleType',
	'Series'
		]]

def clear_use_range(wb, sheet_name):
	wb.sheets(sheet_name).activate()
	active_sheet = xw.sheets.active
	used_range_rows = (active_sheet.api.UsedRange.Row, 
		active_sheet.api.UsedRange.Row + active_sheet.api.UsedRange.Rows.Count)
	used_range_cols = (active_sheet.api.UsedRange.Column, 
		active_sheet.api.UsedRange.Column + active_sheet.api.UsedRange.Columns.Count)
	used_range = xw.Range(*zip(used_range_rows, used_range_cols))
	used_range.clear()

# out_wb = xw.books.open(r"X:\EPA_MPG\not_matched_comparator.xlsm")
# clear_use_range(out_wb, 'not_matched')
# out_wb.sheets('not_matched').range(1, 1).value = not_matched_out
# clear_use_range(out_wb, 'vin_only')
# out_wb.sheets('vin_only').range(1, 1).value = not_matched_vins_for_comparison

# Create the ranges of values for each VIN. 
matched_vins_ranges = matched_vins_simple.groupby('VIN')
matched_vins_ranges['highway08', 'comb08', 'city08'].describe(percentiles=[]).unstack().reset_index().to_csv('duplicate_ranges.csv')

from datetime import datetime
col_names = 'vin_id, vid_date, vin, mm2, vin8, vin1012, vtyp3'.split(', ')
col_types = [np.object_, datetime, np.object_, np.int64, np.object_, np.object_, np.object_]
col_typedict = dict(zip(col_names, col_types))
vtype = pd.read_csv('for_yb.csv', names=col_names, dtype=col_typedict, usecols=['vin', 'vin8', 'vin1012', 'vtyp3'])

matched_mins = matched_vins_ranges.min().reset_index()
['city08']
matched_mins['vin8'] = matched_mins.VIN.apply(lambda x: x[:8])
matched_mins['vin1012'] = matched_mins.VIN.apply(lambda x: x[9:12])
'city08'


vin_original['vin8'] = vin_original.VIN.apply(lambda x: x[:8])
vin_original['vin1012'] = vin_original.VIN.apply(lambda x: x[9:12])

merged_vin_original = pd.merge(vin_original, vtype, on=['vin8', 'vin1012'], how='left')

merged_vin_original.groupby('VIN')

t = merged_vin_original.head(1000)

merged_vin_original.drop_duplicates(subset='VIN').to_csv('vin_with_vtyp3.csv')