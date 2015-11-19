__author__ = 'Thomas'

import datetime
import pandas as pd
import numpy as np
import random
from scipy.stats import expon
from collections import  Counter
import matplotlib.pyplot as plt
import seaborn



def DaysSince(data):
    d1 = datetime.datetime(2013,03,1)
    shapeval = 10
    data = data[datetime.datetime(2013,3,1)-datetime.timedelta(50):'20130301']
    data['days_since'] = [(d1-j).days+1 for j in data.index]
    dsince = [1-expon.cdf(j,scale=shapeval) for j in list(np.unique(data['days_since']))]
    dsince /=np.sum(dsince)
    dsince = np.cumsum(sorted(dsince,reverse=False))
    dsince = (sorted(dsince,reverse=True))
    dlist = sorted(np.unique(data['days_since']),reverse=False)
    dsince /= np.sum(dsince)
    dsince = np.cumsum(dsince)
    u = np.random.uniform(size=10000)
    vals = np.digitize(u,dsince)
    klength = 390+15
    return np.array(data),dlist,dsince,klength
def ExponBoot(data,dlist,dsince,klength):
    '''

    :param data:
    :return: array with bootstrapped resids
    '''
    uninumbers = np.random.uniform(size=(klength,10000))
    t0 = datetime.datetime.now()
    a = np.array([[[j for j,i in zip(dlist,dsince) if uninumbers[k,q]<=i][0] for k in range(klength)] for q in range(10000)])
    print datetime.datetime.now()-t0
    t0 = datetime.datetime.now()
    b = np.array([[random.choice(np.extract(data[:,-1]==i,data)) for i in a[q,:]] for q in range(10000)])
    print datetime.datetime.now()-t0
    print b.shape
    exit()
    return b

def Test(data):
    shockMatrix = np.array([random.choice(data.values) for s in range(len('JA')+403)])
    return shockMatrix

if __name__ == "__main__":
    data = pd.read_csv('data/minutedata4.csv',index_col=0)
    data.index = pd.to_datetime(data.index)
    data = np.log(data).diff().dropna()
    data,dlist,dsince,klength = DaysSince(data)
    for i in range(1000):
        t0 = datetime.datetime.now()
        t = ExponBoot(data,dlist,dsince,klength)
        #t = Test(data)
        print datetime.datetime.now()-t0

