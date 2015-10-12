#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

t0 = time.time()

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)


def EstimateVAR(data, H):
    """

    :param data: A numpy array of log returns
    :param H: integer, size of step ahead forecast
    :return: a dataframe of connectivity or concentration parameters
    """

    model = sm.VAR(data)
    results = model.fit(maxlags=10, ic='aic')

    SIGMA = np.cov(results.resid.T)
    ma_rep = results.ma_rep(maxn=H)
    GVD = np.zeros_like(SIGMA)

    r, c = GVD.shape
    for i in range(r):
        for j in range(c):
            GVD[i, j] = 1 / np.sqrt(SIGMA[i, i]) * sum([ma_rep[h, i].dot(SIGMA[j]) ** 2 for h in range(H)]) / sum(
                [ma_rep[h, i, :].dot(SIGMA).dot(ma_rep[h, i, :]) for h in range(H)])
            # GVD[i,j] = SIGMAINV[i,i] * sum([ma_rep[h,i].dot(SIGMA[j])**2 for h in range(H)]) / sum([ma_rep[h,i,:].dot(SIGMA).dot(ma_rep[h,i,:]) for h in range(H)])
        GVD[i] /= GVD[i].sum()

    return pd.DataFrame(GVD), SIGMA, ma_rep, results.resid.T


def BootstrapMult(resid, marep, iter):
    '''

    Ikke færdiggjort.
    Funktionene skal replikere "iter" perioders afkast af "periods" længde ved at bootstrappe shockvektorer fra
    "resid"

    :param resid:
    :param marep:
    :param iter:
    :return:
    '''

    # Number of periods to simulate, and length of the response to shocks
    periods = int(60 * 7.5)  # en dag i minutter
    responseLength = len(marep)
    nAssets = len(marep[0])

    dailyReturns = []

    for i in range(iter):
        simReturns = np.zeros((periods, nAssets))
        simValues = np.ones((periods + 1, nAssets))

        shockMatrix = np.array([random.choice(resid.T.values) for x in range(len(simReturns) + 15)])

        impulseResponseSystem = marep[::-1]  # Invert impulse responses to fit DataFrame
        for t in range(len(simReturns)):
            for h in range(responseLength):
                simReturns[t] += impulseResponseSystem[h].dot(shockMatrix[t + h - responseLength + 1])
            simValues[t + 1] *= simValues[t] * (simReturns[t] + 1)

        dailyReturns.append(simValues[-1, :].sum() / simValues.shape[1])

    return dailyReturns

def realizedDaily():
    df = pd.read_csv('data/dailyData.csv', sep=",")
    df = df.ix[:, 2:]
    df = np.log(df).diff().dropna()+1
    return (df.sum(axis=1) / len(df.columns)).values

if __name__ == "__main__":

    df = pd.read_csv('data/CRSP_IndexData.csv', sep=",", nrows=100000)
    print "data loaded", time.time() - t0
    df = df.set_index(pd.to_datetime(df['DATE'] + ' ' + df['TIME']))
    df = df.ix[:, 2:]
    df = df.asfreq('1Min')
    df = np.log(df).diff().dropna()

    actualReturns = realizedDaily()
    con, sigma, marep, resid = EstimateVAR(df, 15)
    modelReturns = BootstrapMult(resid, marep, 1000)


    sns.distplot(modelReturns, norm_hist=True,label="Model")
    sns.distplot(actualReturns, norm_hist=True,label="Actual")
    sns.distplot(np.random.normal(np.mean(modelReturns),np.std(modelReturns),len(modelReturns)),norm_hist=True,label="Normal")
    sns.distplot(np.random.normal(np.mean(actualReturns),np.std(actualReturns),len(actualReturns)),norm_hist=True,label="Normal2")
    print time.time()-t0
    plt.legend()
    plt.show()
    exit()
    Bootstrap1p(sigma, 100)
    exit()
    df10 = pd.rolling_apply(df, 10, lambda x: np.prod(1 + x) - 1)
    df10 = df10.dropna().values.flatten()
    df10 = df10 - np.mean(df10) + 1
    print a.shape
    print b.shape
    sns.distplot(a, label='i.i.d.', norm_hist=True, bins=500)
    sns.distplot(b, label='Sequential', norm_hist=True, bins=500)
    sns.distplot(df10, label='Historical', norm_hist=True, bins=500)
    plt.legend()
    plt.show()
