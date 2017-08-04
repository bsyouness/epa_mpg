import xlwings as xw
import pandas as pd
import re
import operator
from datetime import datetime
from math import isnan
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def load_data(vin_file=r'X:\EPA_MPG\vin_with_vtyp3.csv', epa_file=r'X:\EPA_MPG\epa_data.csv', init_yr=1991, last_yr=np.inf):
	global epa_raw, vin_raw
	
	epa_raw = pd.read_csv(epa_file, encoding='utf8', dtype=unicode)
	vin_raw = pd.read_csv(vin_file, encoding='utf8', dtype=unicode)

	epa_original = epa_raw.copy()
	vin_original = vin_raw.copy()

	# Fix errors in the datasets. 
	vin_original.loc[(vin_original.ModelYear == '1998') & (vin_original.Make == 'FORD') & (vin_original.Model == 'Expedition')
		& (vin_original.DisplacementL == u'14.6'), ['DisplacementL', 'FuelTypePrimary']] = ['4.6', 'gasoline']

	vin_original.loc[(vin_original.ModelYear == '1998') & (vin_original.Make == 'FORD') & (vin_original.Model == 'Explorer') 
		& ((vin_original.VIN.apply(lambda x: x[7]) == 'E') | (vin_original.VIN.apply(lambda x: x[7]) == 'X')), 
		['DisplacementL', 'FuelTypePrimary']] = ['4.0', 'gasoline']

	vin_original.loc[(vin_original.ModelYear == '1998') & (vin_original.Make == 'FORD') & (vin_original.Model == 'Explorer') 
		& (vin_original.VIN.apply(lambda x: x[7]) == 'P'), 
		['EngineCylinders', 'DisplacementL', 'FuelTypePrimary']] = ['8', '5.0', 'gasoline']

	# Define integer id based on error code from VIN database. 
	vin_original['error_id'] = vin_original.ErrorCode.apply(
		lambda x: re.match('([0-9]+).*', x).groups()[0])

	# Define columns on which the merge will be performed.
	epa_cols = [
		'make',
		'model',
		'year',
		'fuelType1',
		'fuelType2',
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
		'FuelTypeSecondary',
		'DriveType',
		'TransmissionStyle', 
		'TransmissionSpeeds',
		'EngineCylinders',
		'DisplacementL',
		]

	# Get rid of undesirable columns.
	vin_keep_cols = ['VIN', 'VehicleType', 'BodyClass', 'error_id', 'Series', 'vtyp3', 'counts', 'Trim', 'Trim2'] + vin_cols 
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
	filter_out_strs = 'incomplete vehicle, trailer, motorcycle, bus, low speed vehicle (lsv)'.split(', ')
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

	# Drop certain makes. 
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

	# Take out flexible fuel vehicles. 
	vin_original, epa_original = [df.loc[~(df.fuelType1.str.contains('ffv|flexible|ethanol|e85|natural gas')) &
		~(df.model.str.contains('ffv|flexible|ethanol|e85|natural gas'))] 
		for df in (vin_original, epa_original)]

	# Modify fuel types to account for electric vehicles. 
	def mod_electric_vehicles(df):
		df.loc[df.model.str.contains('plug'), 'fuelType1'] = 'phev'
		df.loc[(df.model.str.contains('(hev|hybrid)')) | 
			((df.fuelType1.str.contains('gasoline')) & (df.fuelType1.str.contains('electric'))) |
			((df.fuelType1.str.contains('gasoline')) & (df.fuelType2.str.contains('electric'))), 
			'fuelType1'] = 'hev'
		df.loc[(df.model.str.contains('bev')) | (df.model.str.contains('electric')) | 
			(df.fuelType1.str.contains('electric')), 'fuelType1'] = 'bev'
		return df
	epa_original = mod_electric_vehicles(epa_original)
	vin_original = mod_electric_vehicles(vin_original)

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
			df[item + '_mod'] = df[item]
			df[item + '_mod'] = df[item].replace(mapping[df_name][item])

	# Make years ints.
	epa_original.year = epa_original.year.apply(int)
	vin_original.year = vin_original.year.apply(int)

	# Only keep years between 1991 and 2010. 
	vin_original = vin_original.loc[(vin_original.year >= 1991)] # & (vin_original.year <= 2010)]
	epa_original = epa_original.loc[(epa_original.year >= 1991)] # & (epa_original.year <= 2010)]

	# Add an ID for EPA before the splitting occurs. Equivalent to VIN. 
	epa_original['EPA'] = range(1, len(epa_original) + 1)

	# Add model_mod.
	vin_original['model_mod'] = vin_original['model']
	epa_original['model_mod'] = epa_original['model']

	# Remove class and series from model names. 
	index_mod = vin_original.loc[vin_original.model_mod.str.contains(r'\w+[ ]*-[ ]*(?:class|series)')].index
	vin_original.loc[index_mod, 'model_mod'] = \
		vin_original.loc[index_mod, 'model_mod'].str.extract(r'(\w+)[ ]*-[ ]*(?:class|series)')

	# Replace all instances of pick-up with pickup.
	vin_original['model_mod'] = vin_original.model_mod.str.replace('pick-up', 'pickup')

	# Turn anything that looks like this: texttext-123123 (\w+-\d+) into texttext123123 (drop the dash)
	# e.g. f-350
	# Do this using a for-loop with df. 
	vin_original.loc[vin_original.model.str.contains(r'[^\d]+-\d+.*'), 'model_mod'], \
	epa_original.loc[epa_original.model.str.contains(r'[^\d]+-\d+.*'), 'model_mod'] = [
		df.loc[df.model.str.contains(r'[^\d]+-\d+.*'), 'model_mod'].str.replace('-', '') 
		for df in (vin_original, epa_original)]

	# For chrysler models, replace town & country with townandcountry. 
	vin_original.loc[(vin_original.make == 'chrysler') & (vin_original.model_mod == 'town & country'), 'model_mod'] = \
		u'townandcountry'

	# Split rows that contain separators into several rows. 
	## In VIN. 
	vin_expanded = pd.concat(
			[pd.Series(np.append(row[[col for col in vin_original.columns if col != 'model_mod']].values, [x]))
				for _, row in vin_original.iterrows() 
				for x in map(unicode.strip, re.findall(r'[\w -]+', row['model_mod']))],
			axis=1).transpose()
	def split_row(s, separator):
		pattern1 = re.compile(r'(.*?)(?=\S*(?:{}) *\S*?)(\S*)(.*)'.format(separator))
		if pattern1.match(s):
			groups = pattern1.match(s).groups()
			parts = map(unicode.strip, re.split(separator, groups[1]))
			# For cases like: srt-8/9
			pattern2 = re.compile(r'([\w\W]*?)(\d+)$') # e.g. srt-9
			subparts = []
			for part in parts:
				if re.match(pattern2, part):
					subparts.append(re.match(pattern2, part).groups())
			# Check that we're in a case like srt-8/9 and not, 636/mrx-8.
			if len(subparts) == len(parts) and subparts[0][0] != '':
				def find_non_blank(t):
					try: return reduce(lambda x, xs: x if x != '' else xs[0], t)
					except: return ''
				default = map(find_non_blank, zip(*subparts))
				parts = [(subpart1 or default[0]) + (subpart2 or default[1]) for (subpart1, subpart2) in subparts]
			return map(''.join, [[groups[0], x, groups[2]] for x in parts])
		else:
			return [s]
	## In EPA. 
	## First, modify the models that will be split. 
	separator = r'/|,'
	# Trim all white spaces from model names that contains slashes with contiguous spaces.
	def trim_slashes(s):
		for sep in separator.split('|'):
			s = re.sub(' *{} *'.format(sep).encode('string-escape'), sep.encode('string-escape'), s)
		return s
	index_mod = epa_original.loc[epa_original.model.str.contains(separator)].index
	epa_original.loc[index_mod, 'model_mod'] = \
		epa_original.loc[index_mod, 'model_mod'].apply(trim_slashes)
	## Delete all spaces from a subset of the makes. 
	makes_no_spaces = 'bmw, buick, cadillac, lexus, subaru, rolls-royce'.split(', ')
	index_mod = epa_original.loc[(epa_original.make.isin(makes_no_spaces)) & 
		(epa_original.model.str.contains(separator))].index
	epa_original.loc[index_mod, 'model_mod'] = \
		epa_original.loc[index_mod, 'model_mod'].replace(' ', '-', regex=True)
	## Lumina model.
	epa_original.loc[epa_original.model.str.contains(r'(?=.*lumina.*)(?=.*apv.*)'), 'model_mod'] = 'lumina apv'
	## Replace spaces with dashes. 
	def replace_spaces(s, pattern, replace_with='-'):
		replace_with_ = lambda s: s.replace(' ', replace_with)
		found_strs = re.findall(pattern, s)
		for _s in found_strs:
			s = re.sub(_s, replace_with_(_s), s)
		return s
	## Monte-carlo model.
	pattern = re.compile('monte carlo')
	index_mod = epa_original.loc[(epa_original.make == 'chevrolet')].index
	epa_original.loc[index_mod, 'model_mod'] = \
		epa_original.loc[index_mod, 'model_mod'].apply(lambda x: replace_spaces(x, pattern, ''))
	## Chrysler models. 
	models_w_spaces = 'new yorker, town and country, fifth avenue, grand \S*'.split(', ')
	pattern = re.compile('(?=(' + '|'.join('{}'.format(x) for x in models_w_spaces) + '))')
	index_mod = epa_original.loc[(epa_original.make == 'chrysler') & epa_original.model_mod.str.contains(pattern)].index
	epa_original.loc[index_mod, 'model_mod'] = \
		epa_original.loc[index_mod, 'model_mod'].apply(lambda x: replace_spaces(x, pattern, ''))
	## Ferrari models. 
	pattern = re.compile('\S* f1')
	index_mod = epa_original.loc[(epa_original.make == 'ferrari') & epa_original.model_mod.str.contains(pattern)].index
	epa_original.loc[index_mod, 'model_mod'] = \
		epa_original.loc[index_mod, 'model_mod'].apply(lambda x: replace_spaces(x, pattern, ''))
	epa_original.loc[epa_original.make == 'ferrari', 'model_mod'] = \
		epa_original.loc[epa_original.make == 'ferrari', 'model_mod'].apply(lambda s: s.replace('ferrari', '').strip())
	## Mercedes models. 
	epa_original.loc[epa_original.make == 'mercedes-benz', 'model_mod'] = \
		epa_original.loc[epa_original.make == 'mercedes-benz', 'model_mod'].apply(lambda s: s.replace('600sel', '600 sel'))
	## Pontiac models. 
	pattern = re.compile('(.*)(trans [^\s/]*)(.*)')
	index_mod = epa_original.loc[(epa_original.make == 'pontiac') & epa_original.model_mod.str.contains(pattern)].index
	epa_original.loc[index_mod, 'model_mod'] = \
		epa_original.loc[index_mod, 'model_mod'].apply(lambda s: re.sub(pattern, r'\1trans\3', s))
	## Expand EPA models. 
	epa_expanded = pd.concat(
			[pd.Series(np.append(row[[col for col in epa_original.columns if col != 'model_mod']].values, [x]))
				for _, row in epa_original.iterrows() 
				for x in split_row(row['model_mod'], separator)],
			axis=1).transpose()
	vin_expanded.columns, epa_expanded.columns = vin_original.columns, epa_original.columns
	vin_original, epa_original = vin_expanded, epa_expanded
	# Delete all the spaces from remaining models. 
	# vin_original.model_mod, epa_original.model_mod = [df.model_mod.replace(' ', '-', regex=True) for df in (vin_original, epa_original)]

	# Add IDs
	vin_original['VIN_ID'] = range(1, len(vin_original) + 1)
	epa_original['EPA_ID'] = range(1, len(epa_original) + 1)

	# Reset index.
	vin_original = vin_original.reset_index(drop=True)
	epa_original = epa_original.reset_index(drop=True)

	return vin_original, epa_original

def list_split_models(vin_original, epa_original):
	######################################################################################################################
	# Create a list of the split models.
	######################################################################################################################
	# Original VIN and EPA data. 
	vin_original.loc[vin_original.model != vin_original.model_mod, 'model, model_mod, make'.split(', ')
		].drop_duplicates().sort_values('model').to_csv('vin_original_changed_models.csv')
	epa_original.loc[epa_original.model != epa_original.model_mod, 'model, model_mod, make'.split(', ')
		].drop_duplicates().sort_values('model').to_csv('epa_original_changed_models.csv')

	# Processed VIN and EPA data. 
	vin.loc[vin.model != vin.model_mod, 'model, model_mod, make'.split(', ')].drop_duplicates().sort_values('model'
		).to_csv('vin_changed_models.csv')
	epa.loc[epa.model != epa.model_mod, 'model, model_mod, make'.split(', ')].drop_duplicates().sort_values('model'
		).to_csv('epa_changed_models.csv')

def mod_1(vin_1, epa_1):
	# Get rid of rows where certain info is missing.
	essential_cols = 'make, model, year, fuelType1, displ, cylinders'.split(', ')
	for col in essential_cols:
		vin_1 = vin_1.loc[~vin_1[col].isnull()]

	# Make the counts integers. 
	vin_1['counts'] = vin_1['counts'].astype(int)

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
		if re.findall(',', s):
			return unicode(round(float(s.split(',')[0]), 1))
		elif s == u'-1':
			return s
		else:
			return unicode(round(float(s), 1))
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

	# Add vtyp3 for epa. 
	epa_1['vtyp3'] = u'-1'
	epa_1.loc[epa_1.VClass.str.contains('pickup') & epa_1.model.str.contains('15|10'), 'vtyp3'] = u'3'
	epa_1.loc[epa_1.VClass.str.contains('pickup') & epa_1.model.str.contains('25|35'), 'vtyp3'] = u'4'

	# Reset index.
	vin_1 = vin_1.reset_index(drop=True)
	epa_1 = epa_1.reset_index(drop=True)

	return vin_1, epa_1

def mod_2(vin, epa):
	# In vin.
	default_type = '0'
	vin['type'] = np.nan
	vin.loc[(vin.vtyp3.str.contains(r'1|2')) | (vin.VehicleType.str.contains('car')), 'type'] = default_type
	ton_dict = {'1/4': '15', '1/2': '15', '3/4': '25', '1': '35'}
	# In model name. 
	vin.loc[(vin.type != default_type) & (vin.model_mod.str.contains(r'(\D|^)(15|25|35)')), 'type'] = \
		vin.model_mod.str.extract(r'(\D|^)(15|25|35)')[1]
	# In series name. 
	vin.loc[(vin.type != default_type) & (vin.Series.str.contains(r'(\D|^)(15|25|35)')), 'type'] = \
		vin.Series.str.extract(r'(\D|^)(15|25|35)')[1]
	vin.loc[(vin.type != default_type) & (vin.Series.str.contains(r'([^\(\s]*)\ston')), 'type'] = \
		vin.Series.str.extract(r'([^\(\s]*)\ston').replace(ton_dict)
	# In series name. 
	vin.loc[(vin.type != default_type) & (vin.Trim.str.contains(r'(\D|^)(15|25|35)')), 'type'] = \
		vin.Trim.str.extract(r'(\D|^)(15|25|35)')[1]
	vin.loc[(vin.type != default_type) & (vin.Trim.str.contains(r'([^\(\s]*)\ston')), 'type'] = \
		vin.Trim.str.extract(r'([^\(\s]*)\ston').replace(ton_dict)
	# Replace nans with the default type string.
	vin.loc[(vin.type.isnull()) | (~vin.type.isin([default_type] + ton_dict.values())), 'type'] = default_type
	# EPA has a separate model called accord wagon; in VIN, BodyClass identifies the wagons; 
	# so we create a separate model in VIN called accord-wagon.
	vin.loc[(vin.make == 'honda') & (vin.model == 'accord') & (vin.BodyClass == 'wagon'), 'model_mod'] = 'accord-wagon'
	epa.loc[(epa.make == 'honda') & (epa.model == 'accord wagon'), 'model_mod'] = 'accord-wagon'
	# EPA has a separate model called matrix, whereas VIN has a model corolla matrix; 
	# so we create a separate model in VIN called matrix.
	vin.loc[(vin.make == 'toyota') & (vin.model == 'corolla matrix'), 'model_mod'] = 'matrix'

	# In epa.
	# In model name.
	epa['type'] = np.nan
	epa.loc[~epa.VClass.str.contains(r'pickup|sport|van'), 'type'] = default_type
	epa.loc[(epa.type != default_type) & (epa.model.str.contains('10|15|25|35')), 'type'] = \
		epa.model.str.extract('(10|15|25|35)').replace({'10': '15'})
	# Replace nans with 'car'
	epa.loc[epa.type.isnull(), 'type'] = default_type

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
	epa.loc[(epa.make == 'acura') & epa.model.str.contains(r'\d\.\d.*'), 'model_mod'] = \
		epa.loc[(epa.make == 'acura') & epa.model.str.contains(r'\d\.\d.*'), 'model_mod'].apply(
			lambda s: re.sub(r'\d\.\d(.*)', r'\1', s))
	## Infiniti. 
	epa.loc[(epa.make == 'infiniti') & epa.model.str.contains(r'(?=.*?)x(?=$|\s)'), 'model_mod'] = \
		epa.loc[(epa.make == 'infiniti') & epa.model.str.contains(r'(?=.*?)x(?=$|\s)'), 'model_mod'].apply(
			lambda s: re.sub(r'(.*?)x(?=$|\s).*', r'\1', s))
	## Nissan. 
	epa.loc[(epa.make == 'nissan'), 'model_mod'] = \
		epa.loc[(epa.make == 'nissan'), 'model_mod'].replace('truck', 'pickup', regex=True)
	epa.loc[epa.model.str.contains('pathfinder armada'), 'model_mod']	= 'armada'
	## BMW. 
	epa.loc[(epa.make == 'bmw'), 'model_mod'] = \
		epa.loc[(epa.make == 'bmw'), 'model_mod'].replace(' convertible', 'c', regex=True)
	## Delete from EPA all models that contain chassis in model name. 
	epa = epa.loc[~epa.model.str.contains('chassis')]
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
	epa.loc[(epa.make == 'chrysler') & (epa.model.str.contains('300')), 'model_mod'] = '300'
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
	### Drop all the letters at the end of a model with a number. 
	vin.loc[(vin.make == 'mercedes-benz') & (vin.model_mod.str.contains(r'.*\d\D+.*')), 'model_mod'], \
	epa.loc[(epa.make == 'mercedes-benz') & (epa.model_mod.str.contains(r'.*\d\D+.*')), 'model_mod'] = \
		[df.loc[(df.make == 'mercedes-benz') & (df.model_mod.str.contains(r'.*\d\D+.*')), 'model_mod'].apply(
			lambda s: re.sub(r'(.*\d)\D+.*', r'\1', s)) for df in (vin, epa)]
	## Toyota. 
	vin.ix[vin.make == 'scion', ['model', 'model_mod']] = \
		vin.ix[vin.make == 'scion', 'model_mod'].apply(lambda x: x.split(' ')[1] if len(x.split(' '))>1 else x)
	epa.ix[epa.model == 'camry solara', 'model_mod'] = 'solara'
	vin.ix[vin.model.str.contains('4-runner'), 'model_mod'] = '4runner'
	epa.loc[(epa.make == 'toyota'), 'model_mod'] = \
		epa.loc[(epa.make == 'toyota'), 'model_mod'].replace('truck', 'pickup', regex=True)
	vin.loc[(vin.make == 'toyota') & (vin.model == 'camry') & (vin.year == 2002) & (vin.displ_mod == '2.2'),
		'displ_mod'] = '2.4'
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

	# Keep only first word of the model. 
	epa['model_mod'], vin['model_mod'] = [
		df['model_mod'].apply(lambda x: x.strip().split(' ')[0].split('-')[0]) for df in epa, vin] 

	# Reset index.
	vin = vin.reset_index(drop=True)
	epa = epa.reset_index(drop=True)

	return vin, epa

def comb_list(l):
	out = []
	for n in range(len(l), 0, -1):
		combs = combinations(l, n)
		out += map(list, combs)
	return out

def merge(vin, epa, n_var_left=4, keep_epa_models=False):
	"""
	Args:
		n_var_left: number of variables to keep at the top of the list for the merge (variables that won't be dropped). 
	"""
	no_missing_val_cols = [
		'make',
		'model_mod',
		'year',
		'fuelType1_mod',
		'type',
		'drive_mod',
		]

	missing_val_cols = [
		'displ_mod',
		'cylinders',
		'transmission_type_mod',
		'transmission_speeds_mod',
		]

	match_col_list = comb_list(missing_val_cols)
	matched_vins = pd.DataFrame(columns=list(set(vin.columns.tolist() + epa.columns.tolist())) + ['matched_on'])
	remaining_epas = epa if keep_epa_models else None
	for col_list in match_col_list:
		on_cols = no_missing_val_cols + col_list
		remaining_vins = vin.loc[~vin.VIN_ID.isin(matched_vins.VIN_ID)]
		# Make sure we don't match on the missing values. 
		remaining_vins = remaining_vins.loc[reduce(operator.and_, [(remaining_vins[x] != '-1') for x in on_cols])]
		if not keep_epa_models:
			remaining_epas = epa.loc[~epa.EPA_ID.isin(matched_vins.EPA_ID)]
			remaining_epas = remaining_epas.loc[reduce(operator.and_, [(remaining_epas[x] != '-1') for x in on_cols])]
		inner_join = pd.merge(remaining_vins.reset_index(), remaining_epas, how='inner', on=on_cols).set_index('index')
		inner_join['matched_on'] = len(on_cols)
		matched_vins = pd.concat([inner_join, matched_vins])

	on_cols = no_missing_val_cols
	for _ in range(len(no_missing_val_cols) - n_var_left):
		on_cols = on_cols[:-1]
		remaining_vins = vin.loc[~vin.VIN_ID.isin(matched_vins.VIN_ID)]
		if not keep_epa_models:
			remaining_epas = epa.loc[~epa.EPA_ID.isin(matched_vins.EPA_ID)]
		inner_join = pd.merge(remaining_vins.reset_index(), remaining_epas, how='inner', on=on_cols).set_index('index')
		inner_join['matched_on'] = len(on_cols)
		matched_vins = pd.concat([inner_join, matched_vins])

	return matched_vins

def missing_cyl_or_displ():
	global vin, epa, matched_vins_simple

	missing_data_VIN_ID = vin.loc[((vin.displ_mod == "-1") | (vin.cylinders == "-1")) & (vin.counts >= 500), 'VIN_ID']
	missing_data_check = matched_vins_simple.loc[matched_vins_simple.VIN_ID_vin.isin(missing_data_VIN_ID)]
	missing_data_check.to_csv('missing_data_check.csv')
	select_cols = 'VIN_ID_vin, displ_mod_vin, displ_mod_epa, cylinders_vin, cylinders_epa'.split(', ')
	missing_data_no_dupes = missing_data_check.drop_duplicates(subset=select_cols)
	manual = pd.read_csv('manual_cylinders_or_displacement.csv')
	# manual = manual[:432]
	data_comp = pd.merge(missing_data_no_dupes, manual, left_on='VIN_vin', right_on='VIN', how='left')
	out_cols = select_cols + 'VIN_vin, displ_mod, cylinders, counts, NEW DISPL, NEW CYL, Notes'.split(', ')
	data_comp[out_cols].to_csv('missing_cyl_displ_comparison.csv')

def test_merging_performance():
	global vin, epa, vin_vin, epa_epa

	# Testing for n_var_left. 
	n_range = range(6, 0, -1)
	sensitivity_to_n = pd.DataFrame(columns='duplication_rate, merge_fraction_wgt'.split(', '), index=n_range)

	for n in n_range:
		n_matched_vins = merge(vin, epa, n)
		n_matched_vins_simple = pd.merge(pd.merge(n_matched_vins, vin_vin, left_on='VIN_ID', 
			right_on='VIN_ID_vin', how='left'), epa_epa, left_on='EPA_ID', right_on='EPA_ID_epa', how='left')
		n_matched_vins_no_dupes = n_matched_vins_simple.drop_duplicates(subset='VIN_vin')
		n_vins_no_dupes = vin.drop_duplicates(subset='VIN')
		merge_fraction_wgt = float(n_matched_vins_no_dupes['counts_vin'].sum())/n_vins_no_dupes['counts'].sum()
		duplication_rate = float(len(n_matched_vins))/float(len(n_matched_vins_no_dupes))
		print('Merge fraction weighted: {:.2%}'.format(merge_fraction_wgt))
		print('Duplication rate: {:.4}'.format(duplication_rate))
		sensitivity_to_n.loc[n, 'duplication_rate'] = duplication_rate
		sensitivity_to_n.loc[n, 'merge_fraction_wgt'] = merge_fraction_wgt

def get_ignore():
	global vin
	# Create an ignore list for the vin dataset. 
	## Turn type into int. 
	not_matched_vins = vin.loc[~vin.VIN.isin(vins_matched)]
	not_matched_vins.type = not_matched_vins.type.apply(int)
	ignore_vins = pd.Series()
	ignore_vins = pd.concat([ignore_vins, 
		not_matched_vins.loc[
			(not_matched_vins.make == 'ford') & 
			(not_matched_vins.model_mod.str.contains('f250|f350|e150|e250|e350|excursion')), 'VIN']])
	ignore_vins = pd.concat([ignore_vins, 
		not_matched_vins.loc[(not_matched_vins.make == 'dodge') & (not_matched_vins.type >= 15), 'VIN']])
	ignore_vins = pd.concat([ignore_vins, 
		not_matched_vins.loc[(not_matched_vins.make == 'chevrolet') & (not_matched_vins.type >= 25), 'VIN']])

	return ignore_vins

def clear_use_range(wb, sheet_name):
	wb.sheets(sheet_name).activate()
	active_sheet = xw.sheets.active
	used_range_rows = (active_sheet.api.UsedRange.Row, 
		active_sheet.api.UsedRange.Row + active_sheet.api.UsedRange.Rows.Count)
	used_range_cols = (active_sheet.api.UsedRange.Column, 
		active_sheet.api.UsedRange.Column + active_sheet.api.UsedRange.Columns.Count)
	used_range = xw.Range(*zip(used_range_rows, used_range_cols))
	used_range.clear()

def gen_missing_mpgs():
	# Missing fuel efficiency data. 
	# * MISSING IN EPA FEG?;
	# IF ERG_MODEL='Sebring' AND EPA_CYL=6 AND CITY08=. THEN DO;
	# IF ERG_MY=2001 THEN DO; CITY08=17; HIGHWAY08=25; COMB08=20; END;
	# IF ERG_MY=2003 THEN DO; CITY08=19; HIGHWAY08=25; COMB08=21; END;
	# IF 2004 LE ERG_MY LE 2005 THEN DO; CITY08=18; HIGHWAY08=25; COMB08=21; END;
	# IF 2007 LE ERG_MY LE 2011 THEN DO; CITY08=16; HIGHWAY08=27; COMB08=20; END;
	# END;

	# IF ERG_MODEL='Impala' THEN DO;
	# IF ERG_MY=2006 THEN DO; EPA_CYL=6; EPA_DISP=3.5; CITY08=17; HIGHWAY08=25; COMB08=20; END;
	# IF ERG_MY=2011 AND EPA_CYL=6 AND EPA_DISP=3.5 THEN DO; CITY08=18; HIGHWAY08=29; COMB08=22; END;
	# IF ERG_MY=2011 AND EPA_CYL=6 AND EPA_DISP=3.9 THEN DO; CITY08=17; HIGHWAY08=25; COMB08=20; END;
	# END;

	# add the mpg of models that are missing based on the ones that exist already for the heavy ones. 
	# Based on tonnage, year, van vs. truck. 

	global vin, epa

	heavy_vins = vin.loc[vin.VIN.isin(ignore_vins)]
	heavy_mpgs = pd.merge(heavy_vins, epa, on='make, model'.split(', ')).drop_duplicates(subset='VIN')

	keep_cols = [x for x in epa.columns if re.match('.*08.*', x)]
	on_vars = 'type, year, vtyp3'.split(', ')
	min_lookup = epa.groupby(on_vars)[keep_cols].min().reset_index()

	for _ in range(3):
		heavy_left = heavy_vins.loc[~heavy_vins.VIN.isin(heavy_mpgs.VIN)]
		heavy_mpgs = pd.concat([heavy_mpgs, pd.merge(heavy_left, min_lookup, on=on_vars)])
		on_vars = on_vars[:-1]

	heavy_mpgs = heavy_mpgs.sort_values('comb08', ascending=True).drop_duplicates(subset='VIN')

	# heavy_vins.loc[~heavy_vins.VIN.isin(heavy_mpgs.VIN)].make.unique()
	heavy_mpgs.to_csv('heavy.csv')

def create_plots():
	global matched_vins_groups

	matched_mins = matched_vins_groups.min()
	grouped = matched_mins.groupby('vtyp3_vin')

	# vtypes = '2door, 4door, small_pickup, large_pickup, suv, cuv, minivan, full_van'.split(', ')
	# vtypes_lookup = map(lambda x: [str(x)], range(1, 9))
	# vtypes_lookup[1] = '2, 2.1, 2.2'.split(', ')

	vtypes = 'car, pickup, suv/cuv, van'.split(', ')
	vtypes_lookup = ['1, 2'.split(', ')] + ['3, 4'.split(', ')] + ['5, 6'.split(', ')] + ['7, 8'.split(', ')]

	vtypes_dict = dict(zip(vtypes, vtypes_lookup))
	matched_dict = {
		df_name: pd.concat([grouped.get_group(k) for k in key_list], axis=0)
		for df_name, key_list in vtypes_dict.items()
		}

	bins = 100
	mpg = 'comb08_epa'

	fig = plt.figure()
	fig.suptitle('Cumulative distribution of vehicle MPGs for {}'.format(mpg), fontsize=14, fontweight='bold')
	ax = fig.add_subplot(111)
	# fig.subplots_adjust(top=0.85)
	# ax.set_title('axes title')

	# evaluate the histogram
	values, base = np.histogram(matched_mins[mpg], bins=bins)
	#evaluate the cumulative
	cumulative = np.cumsum(values) / float(len(matched_mins))
	# plot the cumulative function
	ax.plot(base[:-1], cumulative, label='total')

	for k, v in matched_dict.items():
		# evaluate the histogram
		values, base = np.histogram(v[mpg], bins=bins)
		#evaluate the cumulative
		cumulative = np.cumsum(values) / float(len(v))
		# plot the cumulative function
		ax.plot(base[:-1], cumulative, label=k)

	# manipulate
	vals = ax.get_yticks()
	ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

	ax.set_xlabel('MPG')
	ax.set_ylabel('Cumulative distribution')

	ax.legend()
	plt.show()

	############################################################
	bins = 100
	fig = plt.figure()
	fig.suptitle('Cumulative distribution of vehicle MPGs', fontsize=14, fontweight='bold')
	ax = fig.add_subplot(111)

	for mpg in 'comb08, highway08, city08'.split(', '):
		# evaluate the histogram
		values, base = np.histogram(matched_mins[mpg], bins=bins)
		#evaluate the cumulative
		cumulative = np.cumsum(values) / float(len(matched_mins))
		# plot the cumulative function
		ax.plot(base[:-1], cumulative, label=mpg)

	# manipulate
	vals = ax.get_yticks()
	ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

	ax.set_xlabel('MPG')
	ax.set_ylabel('Cumulative distribution')

	ax.legend()
	plt.show()

def main(export=False):
	######################################################################################################################
	# Load and modify datasets.
	######################################################################################################################
	vin_original, epa_original = load_data()

	# Modify the basics.
	vin_1 = vin_original.copy()
	epa_1 = epa_original.copy()
	vin_1, epa_1 = mod_1(vin_1, epa_1)

	# Modify makes, models, etc.
	vin = vin_1.copy()
	epa = epa_1.copy()
	vin, epa = mod_2(vin, epa)

	######################################################################################################################
	# Merge datasets. 
	######################################################################################################################
	matched_vins = merge(vin, epa, 3, keep_epa_models=True)
	matched_vins_ids = matched_vins[['VIN_ID', 'EPA_ID', 'matched_on']]
	# Rename columns so the result of the merge is more comprehensible. 
	vin_vin = vin.rename(columns = dict(zip(vin.columns, vin.columns + '_vin')))
	epa_epa = epa.rename(columns = dict(zip(epa.columns, epa.columns + '_epa')))
	# Create a simplifed version of the merged dataset.
	matched_vins_simple = pd.merge(pd.merge(matched_vins_ids, vin_vin, left_on='VIN_ID', right_on='VIN_ID_vin', how='left'), 
		epa_epa, left_on='EPA_ID', right_on='EPA_ID_epa', how='left')
	matched_vins_simple.drop(['EPA_ID', 'VIN_ID'], axis=1, inplace=True)
	matched_vins_no_dupes = matched_vins_simple.sort_values(['comb08_epa', 'transmission_type_mod_vin'], ascending=True
		).drop_duplicates(subset='VIN_vin')
	vins_no_dupes = vin_vin.drop_duplicates(subset='VIN_vin')
	print('Merge fraction weighted: {:.2%}'.format(float(matched_vins_no_dupes['counts_vin'].sum())/
		vins_no_dupes['counts_vin'].sum()))
	print('Duplication rate: {:.4}'.format(float(len(matched_vins_simple))/float(len(matched_vins_no_dupes))))

	######################################################################################################################
	# Generate output files. 
	######################################################################################################################
	# Create simple matched file. 
	if export:
		matched_vins_simple.to_csv('matched_vins.csv', encoding = 'utf8')

	# Find the VINs that weren't matched. 
	vins_matched = matched_vins_simple.VIN_vin
	epas_matched = matched_vins_simple.EPA_epa
	not_matched_vins_ = vin_vin.loc[~vin_vin.VIN_vin.isin(vins_matched)]
	not_matched_epas_ = epa_epa.loc[~epa_epa.EPA_epa.isin(epas_matched)]
	not_matched = pd.concat([not_matched_epas_, not_matched_vins_])

	if export:
		# Create a dataset with all records and export it.
		all_records = pd.concat([not_matched, matched_vins_simple])
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
		# Create csv files. 
		all_records[ordered_cols].to_csv('all_records.csv', encoding='utf8')
		all_records_no_dupes[ordered_cols].to_csv('all_records_no_dupes.csv', encoding='utf8')

	# Get a list of the models to ignore because they are too heavy. 
	ignore_vins = get_ignore()

	# Create a file to track the records that weren't matched. 
	not_matched_vins_for_comparison = vin.loc[~vin.VIN.isin(pd.concat([vins_matched, ignore_vins]))]
	not_matched_epas_for_comparison = epa.loc[~epa.EPA.isin(epas_matched)]
	not_matched_for_comparison = pd.concat([not_matched_epas_for_comparison, 
		not_matched_vins_for_comparison])

	# Matching rate. 
	print('Merge fraction weighted: {:.2%}'.format(
		float(vins_no_dupes.counts_vin.sum() - not_matched_vins_for_comparison.counts.sum())/
		vins_no_dupes.counts_vin.sum()))
	
	# Select subset of columns for export. 
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
		'Trim',
		'VehicleType',
		'Series', 
		'type',
		'vtyp3',
			]]

	out_wb = xw.Book(r"X:\EPA_MPG\not_matched_comparator.xlsm")
	clear_use_range(out_wb, 'not_matched')
	out_wb.sheets('not_matched').range(1, 1).value = not_matched_out
	clear_use_range(out_wb, 'vin_only')
	out_wb.sheets('vin_only').range(1, 1).value = not_matched_vins_for_comparison

	# Generate data for records missing fuel efficiency data. 
	gen_missing_mpgs()
	
	######################################################################################################################
	# MPG range distribution.
	######################################################################################################################
	# Create the ranges of values for each VIN. 
	matched_vins_groups = matched_vins_simple.groupby('VIN_ID_vin')
	matched_vins_ranges = matched_vins_groups['highway08_epa', 'comb08_epa', 'city08_epa'].describe(
		percentiles=[]).unstack().reset_index().set_index('VIN_ID_vin')
	# Get rid of multilevel, and rename columns. 
	matched_vins_ranges.columns = map('_'.join, zip(matched_vins_ranges.columns.get_level_values(0).tolist(),
		matched_vins_ranges.columns.get_level_values(1).tolist()))
	matched_vins_ranges.reset_index(inplace=True)
	# Add counts. 
	matched_vins_ranges = pd.merge(matched_vins_ranges, vin_vin[['VIN_vin', 'VIN_ID_vin', 'counts_vin']])
	matched_vins_ranges_out = pd.merge(matched_vins_ranges, matched_vins_no_dupes[[
		'VIN_ID_vin', 'VIN_vin', 'year_vin', 'make_vin', 'model_vin', 'model_epa', 'fuelType1_vin', 'fuelType1_epa',
		'displ_mod_vin', 'displ_mod_epa', 'cylinders_vin', 'cylinders_epa']], 
		on = 'VIN_vin')

	matched_vins_ranges_out['comb08_max_min%'] = matched_vins_ranges_out['comb08_epa_max'] / matched_vins_ranges_out['comb08_epa_min']
	matched_vins_ranges_out.loc[(matched_vins_ranges_out['comb08_max_min%'] >= 1.20) & (matched_vins_ranges_out['counts_vin'] >= 500)
		].to_csv('large_spreads_and_counts.csv')

	######################################################################################################################
	# Spread distribution.
	######################################################################################################################
	mpgs = 'highway08, comb08, city08'.split(', ')
	def add_end(l, s):
		return map(''.join, zip(l, [s]*len(l)))
	mpgs_epa = add_end(mpgs, '_epa')
	mpgs_epa_max = add_end(mpgs_epa, '_max')
	mpgs_epa_min = add_end(mpgs_epa, '_min')
	spread = pd.DataFrame()
	for mpg, mx, mn in zip(mpgs, mpgs_epa_max, mpgs_epa_min):
		matched_vins_ranges[mpg + '_spread'] = matched_vins_ranges[mx] - matched_vins_ranges[mn]
		print matched_vins_ranges[mpg + '_spread'].value_counts().sort_index()
		print matched_vins_ranges[mpg + '_spread'].value_counts().sort_index()/len(matched_vins_ranges) * 100
		spread[mpg + '_spread_no_wgt'] = matched_vins_ranges[mpg + '_spread'].value_counts().sort_index()	
		spread[mpg + '_spread'] = matched_vins_ranges[[mpg + '_spread', 'counts_vin']].groupby(mpg + '_spread').sum()
		spread[mpg + '_spread%'] = spread[mpg + '_spread']/spread[mpg + '_spread'].sum()
		spread[mpg + '_spread%_cum'] = spread[mpg + '_spread%'].cumsum()
	spread.to_csv('spread.csv') 
	matched_vins_ranges.to_csv('duplicate_ranges.csv')

	######################################################################################################################
	# Summary.
	######################################################################################################################
	# Create a summary.
	vin_out = matched_vins_no_dupes[vin_vin.columns.tolist() + ['matched_on']]
	keep_cols = ['VIN_ID_vin', 'highway08_epa_min', 'comb08_epa_min', 'city08_epa_min', 'highway08_spread',
		'city08_epa_count']
	summary = pd.merge(vin_out, matched_vins_ranges[keep_cols], on='VIN_ID_vin')
	summary.rename(columns={'city08_epa_count': 'duplicate_counts'}, inplace=True)
	summary.to_csv('summary_2.csv')
	print('Unique matches: {:.2%}'.format(
		float(sum(summary.loc[summary.duplicate_counts == 1, 'counts_vin']))/sum(summary.counts_vin)
		))

	freq_dist = summary.duplicate_counts.value_counts(normalize=True).sort_index()

if __name__ == '__main__':
	main()
