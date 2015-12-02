import os
import pandas as pd
import datetime
import time
import numpy as np

print "Start"
flist = sorted(os.listdir('Z:/TAQ/TAQHDF5/'))
slist = os.listdir('data/taq/')

for ff in flist:
    print ff
    if ff.replace('taq_', '')[:4] >= '2001' and ff.replace('taq_', '')[:4] < '2014' and str(ff).replace('taq_','').replace('h5','csv') not in slist:
        slist = os.listdir('data/taq/')
        print "Downloading..."
        t0 = datetime.datetime.now()
        #ff = 'taq_20131231.h5'
        path = "Z:/TAQ/TAQHDF5/" + ff
        df = pd.read_hdf(path, 'Trades')
        ind = pd.read_hdf(path, 'TradeIndex')
        ind['end'] = np.cumsum(ind['count'])
        symlist = 'AAPL AXP BA CAT CSCO CVX DD DIS GE GS HD IBM INTC JNJ JOM KO MCD MMM MRK MSFT NKE PFE PG TRV UNH UTX V VZ WMT XOM'.split(
            ' ')
        ind['ticker'] = [str(j).strip() for j in ind['ticker']]
        ind = ind[ind['ticker'].isin(symlist)].reset_index(drop=True)
        ran = np.array([range(start, end) for start, end in zip(ind['start'], ind['end'])])
        ran = [item for sublist in ran for item in sublist]
        df = df[df.index.isin(ran)]
        df['time'] = pd.to_datetime(df['utcsec'], unit='s')
        for i in ind.index:
            start = int(ind.loc[i, 'start'])
            end = int(ind.loc[i, 'end'])
            df.loc[start:end, 'sym'] = ind.loc[i, 'ticker']
        df.to_csv('data/taq/' + ff.replace('taq_', '').replace('.h5', '') + '.csv', columns=['time', 'price', 'sym'],
                  index=False)
        print datetime.datetime.now() - t0
