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
import datetime


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
    periods = int(60 * 6.5)  # en dag i minutter
    responseLength = len(marep)
    nAssets = len(marep[0])

    dailyReturns = []


    for i in range(iter):
        #t0 = datetime.datetime.now()

        simReturns = np.zeros((periods, nAssets))
        simReturns_test = np.zeros((periods, nAssets))
        simValues = np.ones((periods + 1, nAssets))

        shockMatrix = np.array([random.choice(resid.values) for x in range(len(simReturns) + 15)])
        impulseResponseSystem = marep[::-1]  # Invert impulse responses to fit DataFrame

        if dummy==True:
            pseudoReturn = np.product(np.sum(shockMatrix[15:]+1,axis=1)/11)
            dailyReturns.append(pseudoReturn)
        else:
            for t in range(len(simReturns)):
                simReturns[t] = sum([impulseResponseSystem[h].dot(shockMatrix[t + h - responseLength + 1]) for h in range(responseLength)])
                simValues[t + 1] *= simValues[t] * (simReturns[t] + 1)

            dailyReturns.append(simValues[-1, :].sum() / simValues.shape[1])
        #print datetime.datetime.now()-t0
    return dailyReturns

def realizedDaily(day=False):
    df = pd.read_csv('data/dailyData.csv', sep=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = np.log(df).diff().dropna()+1

    if not day:
        return (df.sum(axis=1) / len(df.columns))
    else:
        return (df.sum(axis=1) / len(df.columns))[day]

def mcVar(data,iter):
    data = (1+data).resample('b',how='prod').dropna()
    _returns = data.sum(axis=1)/len(data.columns)

    _mean = _returns.mean()
    _std = _returns.std()

    return np.random.normal(_mean,_std,iter)

def estimateAndBootstrap(df,p,iter,sparse_method=False):
    con, sigma, marep, resid = EstimateVAR(df, p, sparse_method=sparse_method)
    return BootstrapMult(resid, marep, iter)

def rollingEstimates(trainingData,realData,start,end):
    results = pd.DataFrame(columns=['model5','model5break','model1','model1break',"mc1","mc1break",'mc5','mc5break','Real'],index=realData[start:end].index)
    for date in realData[start:end].index:
        print date
        y1 = date - datetime.timedelta(days=365)
        model = estimateAndBootstrap(trainingData[y1:date],15,1000)
        real = realData[date]
        results['model5'].loc[date]=np.percentile(model,5)
        results['model1'].loc[date]=np.percentile(model,1)
        results['model5break'].loc[date]= results['model5'].loc[date] > real
        results['model1break'].loc[date]= results['model1'].loc[date] > real

        mc_model = mcVar(trainingData[y1:date],1000)
        results['mc5'].loc[date]=np.percentile(mc_model,5)
        results['mc1'].loc[date]=np.percentile(mc_model,1)
        results['mc5break'].loc[date]= results['mc5'].loc[date] > real
        results['mc1break'].loc[date]= results['mc1'].loc[date] > real

        results['real'] = real
        print results.loc[date]

    results.to_csv('results_.csv')


if __name__ == "__main__":
    df = pd.read_csv('data/minutedata2.csv', sep=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = np.log(df).diff().dropna()
    print "data loaded", time.time() - t0

    rollingEstimates(df,realizedDaily(),'20150101','20151001')