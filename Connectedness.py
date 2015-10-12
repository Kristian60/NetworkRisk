#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
import matplotlib.pyplot as plt
from sklearn import covariance
import seaborn as sns
import random
import scipy

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


def EstimateVAR_slow():
    df = pd.read_csv('C:/Users/thoru_000/Dropbox/Pers/PyCharmProjects/Speciale/data.csv', sep=";")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna().ffill().set_index('Date')
    data = np.log(df).diff().dropna()

    model = sm.VAR(data)
    results = model.fit(maxlags=5, ic='aic')

    SIGMA = np.cov(results.resid.T)
    ma_rep = results.ma_rep(maxn=10)

    GVD = np.zeros_like(SIGMA)

    r, c = GVD.shape
    for i in range(r):
        for j in range(c):
            sel_j = np.zeros(r)
            sel_j[j] = 1
            sel_i = np.zeros(r)
            sel_i[i] = 1

            AuxSum = 0
            AuxSum_den = 0

            for h in range(10):
                AuxSum += (sel_i.T.dot(ma_rep[h]).dot(SIGMA).dot(sel_j)) ** 2
                AuxSum_den += (sel_i.T.dot(ma_rep[h]).dot(SIGMA).dot(ma_rep[h].T).dot(sel_i))

            GVD[i, j] = (AuxSum * (1 / SIGMA[i, i])) / AuxSum_den

        GVD[i] /= GVD[i].sum()

    pd.DataFrame(GVD).to_csv('GVD.csv', index=False, header=False)


def Bootstrap1p(sigma, iter):
    r = []
    b_r = []
    for i in range(iter):
        if i % (iter / 500.0) == 0:
            print i

        shock = np.array([random.choice(resid.T.values) for x in range(20)])
        p_r = [1] * 10
        for t, A in enumerate(marep[10::-1]):
            p_r *= shock[t, :].dot(marep[t]) + 1
            print p_r
            exit()

        r.append(sum([0.1 * a for a in p_r]))
        draw = random.choice(range(len(df) - 10))
        b_r.append(sum([0.1 * a for a in df.ix[draw, :] + 1]))

    dis = pd.DataFrame(np.array([r, b_r]).T, columns=["sim r", "br"])
    sns.distplot(dis['sim r'], label="sim", norm_hist=True)
    sns.distplot(dis['br'], label="Bootstrap", norm_hist=True)
    plt.legend()
    plt.show()


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
        print i
        simReturns = pd.DataFrame(np.zeros((periods,nAssets)))
        simValues = pd.DataFrame(np.ones((periods,nAssets)))

        shockMatrix = np.array([random.choice(resid.T.values) for x in simReturns.iterrows()])

        impulseResponseSystem = marep[::-1] #Invert impulse responses to fit DataFrame
        for t, r in simReturns.iterrows():
            if t>=0:
                for h in range(responseLength):
                    simReturns.loc[t] += impulseResponseSystem[h].dot(shockMatrix[t+h-responseLength+1])

                if t==0:
                    simValues.loc[t] *= simReturns.loc[t]+1

                else:
                    simValues.loc[t] *= simValues.loc[t-1] * (simReturns.loc[t]+1)

        dailyReturns.append(simValues.iloc[-1,:].sum() / len(simValues.columns))

    sns.distplot(dailyReturns)
    plt.show()
    exit()


if __name__ == "__main__":
    df = pd.read_csv('data/CRSP_IndexData.csv', sep=",", nrows=10000)
    df = df.set_index(pd.to_datetime(df['DATE'] + ' ' + df['TIME']))
    df = df.ix[:, 2:]
    df = df.asfreq('1Min')
    df = np.log(df).diff().dropna()

    con, sigma, marep, resid = EstimateVAR(df, 15)
    a, b = BootstrapMult(resid, marep, 1000)
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
