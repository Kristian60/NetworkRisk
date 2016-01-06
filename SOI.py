__author__ = 'Thomas'

import pandas as pd
import numpy as np
import datetime
import statsmodels.tsa.api as sm


def SOI(days=50):
    H = 15
    data = pd.read_csv('data/TData9313_final6.csv',index_col=0)
    data = np.log(data).diff()[1:]
    data.index = pd.to_datetime(data.index)
    ddate = datetime.datetime(1994,12,27)
    soidf = pd.DataFrame()
    print days
    while ddate<datetime.datetime(2014,1,1):
        datestr2 = ddate.strftime('%Y%m%d')
        datestr1 = (ddate-datetime.timedelta(days)).strftime('%Y%m%d')
        print datestr1,datestr2, "\t",
        td = data[datestr1:datestr2].dropna(axis=1, how='any')
        model = sm.VAR(td)
        results = model.fit(maxlags=H, ic='aic')
        SIGMA = np.cov(results.resid.T)
        _ma_rep = results.ma_rep(maxn=H)
        GVD = np.empty_like(SIGMA)
        r, c = GVD.shape
        for i in range(r):
            for j in range(c):
                GVD[i, j] = 1 / np.sqrt(SIGMA[i, i]) * sum([_ma_rep[h, i].dot(SIGMA[j]) ** 2 for h in range(H)]) / sum(
                    [_ma_rep[h, i, :].dot(SIGMA).dot(_ma_rep[h, i, :]) for h in range(H)])
            GVD[i] /= GVD[i].sum()
        soi = (len(GVD)-np.trace(np.array(GVD)))/len(GVD)
        print soi
        soidf.loc[td.index[-1].strftime("%Y%m%d"),'SOI'] = soi
        soidf.loc[td.index[-1].strftime("%Y%m%d"),'LL'] = int((len(results.params)-1)/len(td.columns))
        ddate += datetime.timedelta(1)
        soidf.to_csv('SOI_%s_days.csv' % (days,),mode='w',header=True)


SOI(100)