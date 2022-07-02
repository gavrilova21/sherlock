#!/usr/bin/python

import time
import warnings
import sys
import os
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from mincemeatpy import mincemeat


warnings.filterwarnings('ignore')


def mapper(c, arr):
    for i in arr:
        yield i, [c, 1]


def reducer(c, arr):
    lst = [0] * 67
    for task, i in arr:
        lst[task] += i
    return lst


if __name__ == '__main__':
    data_path = "./sherlock/"
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    filenames = [item for item in os.listdir(data_path) if item[0] != '.']

    token = RegexpTokenizer(r'\w+')
    data = []
    for fname in filenames:
        with open(os.path.join(data_path, fname), 'r') as f:
            data.append(token.tokenize(f.read().lower()))
    
    server = mincemeat.Server()
    server.mapfn = mapper
    server.reducefn = reducer
    server.datasource = dict(enumerate(data))

    time_start = time.time()
    results = server.run_server(password='changeme')
    
    print('='*30)
    print(f'Duration: {round(time.time() - time_start, 2)} seconds!')
    print('='*30)
    
    res = np.array(np.vstack(results.values()), dtype=int)
    words = list(results.keys())
    
    df = pd.DataFrame(res, columns=filenames, index=words)
    df.to_csv('res.csv', sep=';', header=True, index=True)
