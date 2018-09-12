# -*- coding: utf-8 -*-
from __future__ import division
import re
import csv, codecs, cStringIO
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import simplejson as json
import pandas as pd
import click
from multiprocessing import Pool, cpu_count
import sys
from datetime import datetime
from collections import OrderedDict
from sqlalchemy import create_engine, MetaData, Table, Column
from sqlite3 import OperationalError

import ees.tools.xl.xlwings_tools as xl

db_name = r'X:/EPA_MPG/epa_mpg.sqlite'
engine = create_engine(r"sqlite:///{}".format(db_name), encoding = 'utf-8')

def to_sql(df, table_name, if_exists='replace'):
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)      

# from https://docs.python.org/2/library/csv.html
class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") if type(s) == unicode
            else s for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

def flatten(y):
    # from: https://medium.com/@amirziai/flattening-json-objects-in-python-f5343c794b10#.z1g4uvs4z
    def _flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                _flatten(x[a], name + a + '_')
        elif type(x) is list:
            for i, a in enumerate(x):
                _flatten(a, name + str(i) + '_')
        else:
            out[name[:-1]] = x
    out = {}
    _flatten(y)
    return out

def get_json(vin):
    base_url = 'https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{}?format=json' #&modelyear={}'
    url = base_url.format(vin)

    # based on https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=10.)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        js = json.loads(session.get(url).text)
        return OrderedDict(flatten(js))
    except:
        return {}

def get_data_serial(vin_list):
    base_url = 'https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{}?format=json' #&modelyear={}'
    dict_list = []

    with click.progressbar(vin_list) as gb:
        for vin in gb:
            dict_list.append(get_json(vin))

    df = pd.DataFrame(dict_list)
    df.to_csv('out_serial.csv')

def get_data_parallel():
    p = Pool(cpu_count())
    n = len(vin_list)
    dict_list = []
    for i, js in enumerate(p.imap_unordered(get_json, vin_list), 1):
        dict_list.append(js)
        sys.stderr.write('\rDone {:.3%}'.format(i/n))

    # js_list = p.map(get_json, vin_list) 
    df = pd.DataFrame(dict_list)
    df.to_csv('out_parallel_in_memory.csv')

def get_data_parallel_stream(vin_list, vin_file_path=r'X:\EPA_MPG\data\out_parallel.csv'):
    p = Pool(cpu_count())
    n = len(vin_list)
    with open(vin_file_path, 'w+b') as f:
        wr = UnicodeWriter(f, quoting=csv.QUOTE_ALL)
        wr.writerow(get_json(vin_list[0]).keys())
        for i, js in enumerate(p.imap_unordered(get_json, vin_list), 1):
            wr.writerow(js.values())
            sys.stderr.write('\rDone {:.3%}'.format(i/n))

def take_out_results_string(file_path):
    """ Remove the leading string from the columns names of the output file. 

    Args:
        file_path (str): name of the ouput file

    Example:
        turns u'Results_0_BatteryA' into 'BatteryA'
    """
    book = xl.Book(file_path)
    sheet = book.sheets[0]
    rg0 = sheet.range('A1')
    rg1 = rg0.end('right')
    columns = rg0.resize(1, rg1.column)
    columns.value = [col.split('_')[-1] for col in columns.value]
    book.save()
    book.close() 

# Another way to remove the string 'Results_0_' from the csv file. 
def fix_column_names(vin_df, vin_file_path=r"X:\EPA_MPG\data\out_parallel.csv", change_in_file=False):
    cols_mod = []
    for col in vin_df.columns:
        match = re.search('Results_0_(.+)', col)
        if match:
            cols_mod.append(match.groups()[0])
        else:
            cols_mod.append(col)
    if change_in_file:
        wb = vp.xl.Book(vin_file_path)
        vin_file_name = os.path.basename(vin_file_path).split('.')[0]
        wb.sheets(vin_file_name).range("A1").value = cols_mod
    return cols_mod

vin_file_name = 'out_missing'
vin_file_path = r"X:\EPA_MPG\data\{}.csv".format(vin_file_name)
def fix_vin_output_file(vin_file_path):
    """Remove extra columns from the file that was just read based on the existing SQL table that contains the VINs.
    
    Args:
        vin_file_path (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    vin_df = pd.read_csv(vin_file_path, dtype=unicode, encoding='utf8')
    vin_df.columns = fix_column_names(vin_df)
    # create a session
    Session = sessionmaker(bind=vp.engine)
    session = Session()
    metadata = MetaData()
    vin_table = Table('vin_with_vtyp_no_counts', metadata, autoload=True, autoload_with=engine)
    old_col_names = [description['name'] for description in session.query(vin_table).column_descriptions]
    new_cols = list(set(vin_df.columns) - set(old_vin_df.columns.tolist()))
    vin_df.drop(columns=new_cols, inplace=True)
    return vin_df

def get_counts(put_in_db=True, file_path=None, file_name='txsafe18', if_exists='replace'):
    """Extract info from Tom's VIN file.
    """
    if not file_path:
        file_path = '../data/{}.txt'.format(file_name)
    with open(file_path, 'r') as f:
        vin_lines = f.readlines()
    def parse(s, position):
        try:
            return ''.join(s.split(',')[position].split())
        except:
            return 0
    vins = map(lambda x: parse(x, 0), vin_lines)
    counts = map(lambda x: parse(x, 3), vin_lines)
    counts_with_vins = pd.DataFrame(zip(vins, counts), columns=['VIN', 'counts'])
    if put_in_db:
        to_sql(pd.Series(vins), 'just_vins', if_exists=if_exists)
        to_sql(counts_with_vins, 'counts_with_vins')
    return counts_with_vins

def add_vtyp(vin_file_path, vtyp_file_path=r"X:\EPA_MPG\data\vtyp_no_dupes.csv", if_exists='append'):
    vin_df = pd.read_csv(vin_file_path, dtype=unicode, encoding='utf8')
    # Get rid of rows with errors. 
    vin_df.dropna(subset=['VIN'], inplace=True)
    vin_df['vin8'] = vin_df['VIN'].apply(lambda s: s[:8])
    vin_df['vin1012'] = vin_df['VIN'].apply(lambda s: s[9:12])

    try:
        vtyp_df = pd.read_sql('vtyp_based_on_VIN8_and_VIN1012', engine)
    except:
        vtyp_df = process_vtyp(vtyp_file_path)
    
    vin_df = pd.merge(vin_df, vtyp_df, how='left', on=['vin8', 'vin1012'])
    to_sql(vin_df, 'vin_with_vtyp_no_counts', if_exists=if_exists)

    return vin_df

def process_vtyp(vtyp_file_path):
    """ Process CSV file with VTYP information and push to DB.
    
    Args:
        vtyp_file_path (str): path to CSV file containing VTYP data. 
    
    Returns:
        pd.DataFrame
    """
    vtyp_df = pd.read_csv(vtyp_file_path, dtype=unicode, encoding='utf8')
    vtyp_df['vin8'] = vtyp_df['VIN'].apply(lambda s: s[:8])
    vtyp_df['vin1012'] = vtyp_df['VIN'].apply(lambda s: s[9:12])
    vtyp_df.drop(columns=['VIN'], inplace=True)
    vtyp_df.drop_duplicates(inplace=True)
    vtyp_df.to_sql('vtyp_based_on_VIN8_and_VIN1012', engine, if_exists='replace', index=False)
    return vtyp_df

def load_epa_data(epa_file_path):
    epa_df = pd.read_csv(epa_file_path, dtype=unicode, encoding='utf8')
    to_sql(epa_df, 'raw_epa_data')

def get_vin_data():
    # Replace this when there is an update.
    vin_list = pd.read_sql('just_vins', engine)
    t0 = datetime.now()
    get_data_parallel_stream(vin_list)
    t1 = datetime.now()
    print '\nParallel runtime: {:.3}'.format(t1-t0)

def add_counts():
    vin_df = pd.read_sql('vin_with_vtyp_no_counts', engine)
    counts = pd.read_sql('counts_with_vins', engine)
    to_sql(pd.merge(vin_df, counts, how='left', on=['VIN']),
        'vin_with_vtyp')

def get_vins_not_read():
    vin_df = pd.read_sql('vin_with_vtyp', engine)
    vins = pd.read_sql('just_vins', engine)
    counts_with_vins = pd.read_sql('counts_with_vins', engine)
    not_read = set(vin_df['VIN']) - set(vins)
    not_read_with_counts = pd.merge(pd.DataFrame(list(not_read), columns=['VIN']), counts_with_vins, how='left')
    not_read_with_counts.to_csv(r"X:\EPA_MPG\data\vins_not_read.csv")
    return vin_df, not_read_with_counts

def add_vins_not_read():
    vin_df = pd.read_sql('vin_with_vtyp', engine)
    not_read_df = pd.read_csv('../data/vins_not_read_now_read.csv', encoding='utf-8')
    vtyp_df = pd.read_sql('vtyp_based_on_VIN8_and_VIN1012', engine)
    not_read_df['vin8'] = not_read_df['vin'].apply(lambda s: s[:8])
    not_read_df['vin1012'] = not_read_df['vin'].apply(lambda s: s[9:12])
    not_read_df_with_vtyp = pd.merge(not_read_df, vtyp_df, how='left', on=['vin8', 'vin1012'])

    # Modify columns names so they align. 
    upper_vin_cols = [col.upper() for col in vin_df.columns]
    upper_not_read_cols = [col.upper() for col in not_read_df_with_vtyp.columns]
    missing_cols = set(upper_not_read_cols) - set(upper_vin_cols) 
    added_cols = set(upper_vin_cols) - set(upper_not_read_cols)
    vin_mapping = dict(zip(upper_vin_cols, vin_df.columns))
    not_read_mapping = dict(zip(not_read_df_with_vtyp.columns, upper_not_read_cols))
    col_mapping = dict((k, vin_mapping.setdefault(v, k)) for (k, v) in not_read_mapping.items())

    not_read_df_with_vtyp.rename(columns=col_mapping, inplace=True)
    new_vin_df = pd.concat([vin_df, not_read_df_with_vtyp], axis=0)
    new_vin_df.to_sql('vin_with_vtyp', engine, if_exists='replace', index=False)
    return new_vin_df

if __name__ == "__main__":
    pass