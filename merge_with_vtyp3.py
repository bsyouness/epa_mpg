import pandas as pd
from datetime import datetime
import numpy as np
import re

epa_original = pd.read_csv(r'X:\EPA_MPG\epa_data.csv', encoding='utf8', dtype=unicode)
vin_original = pd.read_csv(r'X:\EPA_MPG\vin_data_original.csv', encoding='utf8', dtype=unicode)

# Get rid of the 'Results_0_' string in the column titles
pattern = re.compile(u'Results_0_(.+)')
vin_original.columns = [pattern.search(x).groups()[0] 
	if pattern.search(x) else x for x in vin_original.columns]

vin_original['vin8'] = vin_original.VIN.apply(lambda x: x[:8])
vin_original['vin1012'] = vin_original.VIN.apply(lambda x: x[9:12])

col_names = 'vin_id, vid_date, vin, mm2, vin8, vin1012, vtyp3'.split(', ')
col_types = [np.object_, datetime, np.object_, np.int64, np.object_, np.object_, np.object_]
col_typedict = dict(zip(col_names, col_types))

vtype = pd.read_csv(r'X:\EPA_MPG\for_yb.csv', names=col_names, 
	dtype=col_typedict, usecols=['vin', 'vin8', 'vin1012', 'vtyp3'])

vtype_no_dupes = vtype.drop_duplicates(subset=['vin8', 'vin1012'])
vtype_no_dupes.to_csv('vtype_no_dupes.csv')

merged = pd.merge(vin_original, vtype_no_dupes,
	how='left'
	).drop_duplicates(subset='VIN')

merged.to_csv('vin_with_vtyp3.csv', encoding='utf8')

# With new file. 
new_vtype = pd.read_csv(r'X:\EPA_MPG\vtyp.csv', dtype=unicode)
# Replace 7.099609375 with 7
new_vtype.loc[new_vtype.VTYP == '7.099609375', 'VTYP'] = '7'
new_vtype.columns = 'VIN, counts, vtyp3'.split(', ')

merged = pd.merge(vin_original, new_vtype,
	how='left', on='VIN'
	).drop_duplicates(subset='VIN')

merged.to_csv('vin_with_vtyp3.csv', encoding='utf8')