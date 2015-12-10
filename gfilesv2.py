__author__ = 'Thomas'

import h5py
import pandas as pd
import datetime
import numpy as np
import os


flist = sorted(os.listdir('Z:/TAQ/TAQHDF5/'))
#slist = os.listdir('data/taq/')

for ff in flist[::-1]:

    if ff.replace('taq_', '')[:4] >= '1990' and ff.replace('taq_', '')[:4] < '2014' and str(ff).replace('taq_','').replace('h5','csv') not in os.listdir('D:/Speciale Data'):
        print ff,
        #slist = os.listdir('data/taq/')
        pd.DataFrame().to_csv('D:/Speciale Data/' + ff.replace('taq_', '').replace('.h5', '') + '.csv')
        t0=datetime.datetime.now()
        f = h5py.File("Z:/TAQ/TAQHDF5/" + ff, 'r')
        symlist = 'AAPL AXP BA CAT CSCO CVX DD DIS GE GS HD IBM INTC JNJ JOM KO MCD MMM MRK MSFT NKE PFE PG TRV UNH UTX V VZ WMT XOM'.split(
            ' ')
        ind = pd.DataFrame(np.array(f['TradeIndex']))
        ind['end'] = np.cumsum(ind['count'])
        ind['ticker'] = [str(j).strip() for j in ind['ticker']]
        ind = ind[ind['ticker'].isin(symlist)].reset_index(drop=True)

        df = []
        for i in ind.index:
            start = int(ind.loc[i,'start'])
            end = int(ind.loc[i,'end'])
            df.extend(f['Trades'][start:end])

        ind['count'] = np.cumsum(ind['count'])
        fr = 0
        df = pd.DataFrame(np.array(df))
        for i in ind.index:
            to = int(ind.loc[i,'count'])
            df.loc[fr:,'sym'] = ind.loc[i,'ticker']
            fr = to

        df['time'] = pd.to_datetime(df['utcsec'], unit='s')
        df.to_csv('D:/Speciale Data/' + ff.replace('taq_', '').replace('.h5', '') + '.csv', columns=['time', 'price', 'sym'],
                          index=False)

        print datetime.datetime.now()-t0