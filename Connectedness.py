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
from statsmodels.tsa.vector_ar.var_model import ma_rep


t0 = time.time()

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)


def EstimateVAR(data, H, sparse_method=False):
    """

    :param data: A numpy array of log returns
    :param H: integer, size of step ahead forecast
    :return: a dataframe of connectivity or concentration parameters
    """

    model = sm.VAR(data)
    results = model.fit(maxlags=H, ic='aic')

    SIGMA = np.cov(results.resid.T)

    if sparse_method==True:
        _nAssets = results.params.shape[1]
        _nLags = results.params.shape[0]/results.params.shape[1]

        custom_params = np.where(abs(results.params / results.stderr)>1.96,results.params,0)[1:].reshape((_nLags,_nAssets,_nAssets))
        _ma_rep = ma_rep(custom_params,maxn=H)
    else:
        _ma_rep = results.ma_rep(maxn=H)

    GVD = np.zeros_like(SIGMA)

    r, c = GVD.shape
    for i in range(r):
        for j in range(c):
            GVD[i, j] = 1 / np.sqrt(SIGMA[i, i]) * sum([_ma_rep[h, i].dot(SIGMA[j]) ** 2 for h in range(H)]) / sum(
                [_ma_rep[h, i, :].dot(SIGMA).dot(_ma_rep[h, i, :]) for h in range(H)])
        GVD[i] /= GVD[i].sum()

    return pd.DataFrame(GVD), SIGMA, _ma_rep, results.resid


def BootstrapMult(resid, marep, iter, dummy=False):
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

        shockMatrix = np.array([random.choice(resid.values) for x in range(len(simReturns) + 15)])
        impulseResponseSystem = marep[::-1]  # Invert impulse responses to fit DataFrame

        if dummy==True:
            pseudoReturn = np.product(np.sum(shockMatrix[15:]+1,axis=1)/11)
            dailyReturns.append(pseudoReturn)
        else:
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

    df = pd.read_csv('data/minutedata.csv', sep=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = np.log(df).diff().dropna()
    print "data loaded", time.time() - t0


    # Actual bootstrapped values
    actualReturns = realizedDaily()
    print "Actual returns constructed", time.time() - t0
    sns.distplot(actualReturns, norm_hist=True, label="Actual")

    # Modeling using sparse method
    con, sigma, marep, resid = EstimateVAR(df, 15, sparse_method=True)
    print "Model Estimation done", time.time() - t0
    modelReturns_Sparse = BootstrapMult(resid, marep, 1000)
    sns.distplot(modelReturns_Sparse, norm_hist=True, label="Sparse")

    # Modeling using full MA-rep
    con, sigma, marep, resid = EstimateVAR(df, 15, sparse_method=False)
    print "Model Estimation done", time.time() - t0
    modelReturns_Full = BootstrapMult(resid, marep, 1000)
    sns.distplot(modelReturns_Full, norm_hist=True, label="Full")

    # Passing through shocks as dummy data
#    modelReturns_dummy = BootstrapMult(resid, marep, 1000, dummy=True)
#    sns.distplot(modelReturns_dummy, norm_hist=True, label="Dummy")

    print time.time()-t0
    plt.legend()
    plt.show()

