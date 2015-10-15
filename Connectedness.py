#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy
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

    if sparse_method == True:
        _nAssets = results.params.shape[1]
        _nLags = results.params.shape[0] / results.params.shape[1]

        custom_params = np.where(abs(results.params / results.stderr) > 1.96, results.params, 0)[1:].reshape(
            (_nLags, _nAssets, _nAssets))
        _ma_rep = ma_rep(custom_params, maxn=H)
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
        # t0 = datetime.datetime.now()

        simReturns = np.zeros((periods, nAssets))
        simReturns_test = np.zeros((periods, nAssets))
        simValues = np.ones((periods + 1, nAssets))

        shockMatrix = np.array([random.choice(resid.values) for x in range(len(simReturns) + 15)])
        impulseResponseSystem = marep[::-1]  # Invert impulse responses to fit DataFrame

        if dummy == True:
            pseudoReturn = np.product(np.sum(shockMatrix[15:] + 1, axis=1) / 11)
            dailyReturns.append(pseudoReturn)
        else:
            for t in range(len(simReturns)):
                simReturns[t] = sum([impulseResponseSystem[h].dot(shockMatrix[t + h - responseLength + 1]) for h in
                                     range(responseLength)])
                simValues[t + 1] *= simValues[t] * (simReturns[t] + 1)

            dailyReturns.append(simValues[-1, :].sum() / simValues.shape[1])
            # print datetime.datetime.now()-t0
    return dailyReturns


def realizedDaily(day=False):
    df = pd.read_csv('data/dailyData.csv', sep=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = np.log(df).diff().dropna() + 1

    if not day:
        return (df.sum(axis=1) / len(df.columns))
    else:
        return (df.sum(axis=1) / len(df.columns))[day]


def mcVar(data, iter):
    data = (1 + data).resample('b', how='prod').dropna()
    _returns = data.sum(axis=1) / len(data.columns)

    _mean = _returns.mean()
    _std = _returns.std()

    return np.random.normal(_mean, _std, iter)


def zeroDeltaVar(data):
    data = (1 + data).resample('b', how='prod').dropna()
    _returns = data.sum(axis=1) / len(data.columns)

    _mean = _returns.mean()
    _std = _returns.std()

    return np.random.normal(_mean, _std, iter)


def estimateAndBootstrap(df, p, iter, sparse_method=False):
    con, sigma, marep, resid = EstimateVAR(df, p, sparse_method=sparse_method)
    return BootstrapMult(resid, marep, iter)


def evaluateModel(VaRs):
    realReturns = realizedDaily()


def formalTests(results, realData):
    def unconditionalCoverage(events, p, t):
        '''
        Simple binomial test of event frequency, Jorion(2001)
        :param events: a pandas series of True/False events of events occuring
        :param p: true likelihood of event
        :param t: length of test
        :return: the p-value of the test
        '''
        x = sum(events)
        return scipy.stats.norm.cdf((x - p * t) / np.sqrt(p * (1 - p) * t))

    def pofTest(events, p, t, raw_output=False):
        '''
        Proportion of faliures test. LR-test that follows a chi2 distribution
        Kupiec(1995)
        :param events: a pandas series of True/False events of events occuring
        :param p: true likelihood of event
        :param t: length of test
        :return: the p-value of the test
        '''
        x = sum(events)
        if raw_output:
            return -2 * np.log(
                (np.power(1 - p, t - x) * np.power(p, x)) / (np.power(1 - (x / t), t - x) * np.power(x / t, x)))
        else:
            return scipy.stats.chi2.cdf(-2 * np.log(
                (np.power(1 - p, t - x) * np.power(p, x)) / (np.power(1 - (x / t), t - x) * np.power(x / t, x))), df=1)

    def tuffTest(events, p, raw_output=False):
        '''
        Time untill first faliure test, suggested by Kupiec(1995)
        :param events: a pandas series of True/False events of events occuring
        :param p: true likelihood of event
        :param v: the first occurence of an event
        :return: the p-value of the test
        '''
        if events.mean == 0:
            return None
        else:
            v = np.argmax(events.values) + 1

            if raw_output:
                return (v,-2 * np.log(np.power(p * (1 - p), v - 1) / ((1 / v) * np.power(1 - (1 / v), v - 1))))
            else:
                return scipy.stats.chi2.cdf(
                    -2 * np.log(np.power(p * (1 - p), v - 1) / ((1 / v) * np.power(1 - (1 / v), v - 1))), df=1)

    def christoffersenIFT(events, p, t):
        '''
        the tuffTest combined with a test of independent events.
        nxx defines the simultaneous probabilities of an event happening after an event/non-event
        pix defines the conditional bayseian probabilities of an event happening

        lastly the independence test is added to the pof-test and evaluated in a chi2 distribution with 2 DOF.

        :param events: a pandas series of True/False events of events occuring
        :param p: true likelihood of event
        :param v: the first occurence of an event
        :return: the p-value of the test
        '''
        _events = pd.concat([events, events.shift()], axis=1).dropna()
        _events.columns = ['e', 'e-1']

        n00 = len(_events[(_events['e'] == False) & (_events['e'] == False)]) / len(_events)
        n10 = len(_events[(_events['e'] == True) & (_events['e'] == False)]) / len(_events)
        n01 = len(_events[(_events['e'] == False) & (_events['e'] == True)]) / len(_events)
        n11 = len(_events[(_events['e'] == True) & (_events['e'] == True)]) / len(_events)

        pi0 = n01 / (n00 + n01)
        pi1 = n11 / (n10 + n11)
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)

        LRind = -2 * np.log((np.power((1 - pi), n00 + n10) * np.power(pi, n01 + n11)) / (
        np.power(1 - pi0, n00) * np.power(pi0, n01) * np.power(1 - pi1, n10) * np.power(pi1, n11)))
        LRpof = pofTest(events, p, t, raw_output=True)

        return scipy.stats.chi2.cdf(LRind + LRpof, 2)

    def mixedKupiecTest(events, p, t):

        LRindependent = 0
        r = tuffTest(events,p,t)
        if r == None:
            return 1
        n, LRtuff = r
        LRindependent += LRtuff

    data = pd.concat([results, realData], axis=1).dropna()
    data['e1'] = data[0] < data['VaR1']
    data['e5'] = data[0] < data['VaR5']

    t = len(data)

    print mixedKupiecTest(data['e1'], 0.01, t)
    print christoffersenIFT(data['e1'], 0.01, t)
    print christoffersenIFT(data['e5'], 0.05, t)
    print pofTest(data['e1'], 0.01, t)
    print pofTest(data['e5'], 0.05, t)
    print unconditionalCoverage(data['e1'], 0.01, t)
    print unconditionalCoverage(data['e5'], 0.05, t)
    print tuffTest(data['e1'], 0.1)
    print tuffTest(data['e5'], 0.5)
    exit()


def backtest(trainingData, realData, start, end, memory, model, *args):
    results = pd.DataFrame(columns=['VaR1', 'VaR5'], index=realData[start:end].index)
    for date in results.index:
        dateMemory = date - datetime.timedelta(days=memory)
        modelSim = model(trainingData[dateMemory:date], *args)
        results.loc[date] = [np.percentile(modelSim, 1), np.percentile(modelSim, 5)]

    formalTests(results, realData)


if __name__ == "__main__":
    df = pd.read_csv('data/minutedata2.csv', sep=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = np.log(df).diff().dropna()
    print "data loaded", time.time() - t0

    backtest(df, realizedDaily(), '20150101', '20150115', 50, estimateAndBootstrap, 10, 10)
    print estimateAndBootstrap(df, 10, 10)
