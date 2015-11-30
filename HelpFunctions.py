__author__ = 'Thomas'

import datetime
import pandas as pd
import numpy as np
import random
from scipy.stats import expon
from collections import Counter
import matplotlib.pyplot as plt
import seaborn

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

def bootstrapExpDecayGraph(data, nIterations):
    d1 = data.index[-1]
    d1 = datetime.datetime(d1.year, d1.month, d1.day)

    shapeval = 10
    daysSince = [(d1 - j).days + 1 for j in data.index]

    probDist = [1 - expon.cdf(j, scale=shapeval) for j in np.unique(daysSince)]
    probDist /= np.sum(probDist)
    probDist = np.cumsum(sorted(probDist, reverse=True))

    minsPerDay = data.resample("d", how="count").values[:, 0]
    utilizedLags = int(391 - minsPerDay[0])
    bootstrapLength = 391 + utilizedLags

    print data
    exit()
    data = np.insert(data.values, 0, np.empty_like(data.ix[:utilizedLags, :]), axis=0)

    data = data.reshape((len(data) / 391, 391, data.shape[1]))
    uninumbers = np.random.uniform(size=(bootstrapLength, nIterations))

    a = np.array([np.digitize(uninumbers[_], probDist) for _ in range(bootstrapLength)])

    print a.shape
    exit()

    b = np.array([[random.choice(data[-i, :, :]) for i in a[:, q]] for q in range(nIterations)])
    return b


if __name__ == "__main__":
    data = pd.read_csv('data/minutedata4.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    temp = np.log(data).diff().dropna().resample('d',how='count')
    temp = temp[~temp['CRSPRET'].isin([0,390,391])].index.date

    dd =

    print data
    exit()

    a = data.resample('d',how='count')
    for i in a.index:
        if a.loc[i,'CRSPRET'] not in [0,390,391]:
            print i
    exit()


    print data
    exit()
    b = bootstrapExpDecayGraph(data,10000)

