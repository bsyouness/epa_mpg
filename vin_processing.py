# -*- coding: utf-8 -*-

from __future__ import division
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
from sqlalchemy import create_engine
from sqlite3 import OperationalError

import ees.tools.xl.xlwings_tools as xl

db_name = r'X:/EPA_MPG/epa_mpg.sqlite'
engine = create_engine(r"sqlite:///{}".format(db_name), encoding = 'utf-8')

def to_sql(df, table_name):
    df.to_sql(table_name, engine, if_exists='replace', index=False)      

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
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    js = json.loads(session.get(url).text)
    return OrderedDict(flatten(js))

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

def get_data_parallel_stream(vin_list):
    p = Pool(cpu_count())
    n = len(vin_list)
    with open(r'data\out_parallel.csv', 'wb') as f:
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

def pre_process():
    """Extract info from Tom's VIN file.
    """
    with open('../data/txsafe18.txt', 'r') as f:
        vins_lines = f.readlines()
    def parse(s, position):
        try:
            return ''.join(s.split(',')[position].split())
        except:
            return 0
    vins = map(lambda x: parse(x, 0), vins_lines)
    counts = map(lambda x: parse(x, 3), vins_lines)
    counts_with_vins = pd.DataFrame(zip(vins, counts), columns=['VIN', 'counts'])
    to_sql(pd.Series(vins), 'just_vins')
    to_sql(counts_with_vins, 'counts_with_vins')
    return counts_with_vins

def add_vtyp(vin_file_path, vtyp_file_path=r"X:\EPA_MPG\data\vtyp_no_dupes.csv"):
    vin_df = pd.read_csv(vin_file_path, dtype=unicode, encoding='utf8')
    # Get rid of rows with errors. 
    vin_df.dropna(subset=['VIN'], inplace=True)
    vin_df['vin8'] = vin_df['VIN'].apply(lambda s: s[:8])
    vin_df['vin1012'] = vin_df['VIN'].apply(lambda s: s[9:12])

    try:
        vtyp_df = pd.read_sql('vtyp_based_on_VIN8_and_VIN1012', engine)
    except:
        vtyp_df = process_vtyp(vtyp_file_path)
    
    to_sql(pd.merge(vin_df, vtyp_df, how='left', on=['vin8', 'vin1012']),
        'vin_with_vtyp_no_counts')

def process_vtyp(vtyp_file_path):
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

def vins_not_read():


if __name__ == "__main__":
    pass