#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
import random
import scipy
import time
from statsmodels.tsa.vector_ar.var_model import ma_rep
import datetime
import sys

if hasattr(sys, 'getwindowsversion'):
    import matplotlib.pyplot as plt
    import seaborn as sns

t0 = time.time()

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)


def EstimateVAR(data, H, sparse_method=False, GVD_output=False):
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

    #for x in _ma_rep:
    #    sns.heatmap(x, annot=True)
    #    plt.show()
    #exit()

    GVD = np.zeros_like(SIGMA)

    if GVD_output:
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

    z1 = -scipy.stats.norm.ppf(0.01)
    z5 = -scipy.stats.norm.ppf(0.05)
    return _mean - z1 * _std, _mean - z5 * _std


def estimateAndBootstrap(df, H, iter, sparse_method=False):
    con, sigma, marep, resid = EstimateVAR(df, H, sparse_method=sparse_method)
    df = pd.read_csv('data/dailyData.csv', sep=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = np.log(df).diff().dropna() + 1

    returnSeries = BootstrapMult(resid, marep, iter)
    return np.percentile(returnSeries, 0.01), np.percentile(returnSeries, 0.05)


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
        if events.mean() == 0:
            return None
        else:
            v = np.argmax(events.values) + 1

            if raw_output:
                return (v, -2 * np.log((p * np.power((1 - p), v - 1)) / ((1 / v) * np.power(1 - (1 / v), v - 1))))
            else:
                return scipy.stats.chi2.cdf(
                    -2 * np.log((p * np.power((1 - p), v - 1)) / ((1 / v) * np.power(1 - (1 / v), v - 1))), df=1)

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

        n00 = len(_events[(_events['e'] == False) & (_events['e-1'] == False)])
        n10 = len(_events[(_events['e'] == False) & (_events['e-1'] == True)])
        n01 = len(_events[(_events['e'] == True) & (_events['e-1'] == False)])
        n11 = len(_events[(_events['e'] == True) & (_events['e-1'] == True)])

        try:
            # Conditional probabilities
            pi0 = n01 / (n00 + n01)
            pi1 = n11 / (n10 + n11)

            # Unconditional probability for an event
            pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        except ZeroDivisionError:
            return None

        LRind = -2 * np.log((np.power((1 - pi), n00 + n10) * np.power(pi, n01 + n11)) / (
            np.power(1 - pi0, n00) * np.power(pi0, n01) * np.power(1 - pi1, n10) * np.power(pi1, n11)))
        LRpof = pofTest(events, p, t, raw_output=True)

        return scipy.stats.chi2.cdf(LRind + LRpof, 2)

    def mixedKupiecTest(events, p, t):

        nEvents = sum(events)
        LRind = 0
        for e in range(nEvents):
            n, LRtuff = tuffTest(events, p, raw_output=True)
            LRind += LRtuff
            events = events[n:]

        LRpof = pofTest(events, p, t, raw_output=True)
        return scipy.stats.chi2.cdf(LRind + LRpof, nEvents + 1)

    data = pd.concat([results, realData], axis=1).dropna()
    data['e1'] = data[0] < data['VaR1']
    data['e5'] = data[0] < data['VaR5']

    data.to_csv('Events_' + time.strftime("%Y%m%d", time.gmtime()) + '.csv')

    t = len(data)

    return [unconditionalCoverage(data['e1'], 0.01, t),
            unconditionalCoverage(data['e5'], 0.05, t),
            pofTest(data['e1'], 0.01, t),
            pofTest(data['e5'], 0.05, t),

            tuffTest(data['e1'], 0.1),
            tuffTest(data['e5'], 0.5),

            mixedKupiecTest(data['e1'], 0.01, t),
            mixedKupiecTest(data['e5'], 0.05, t),
            christoffersenIFT(data['e1'], 0.01, t),
            christoffersenIFT(data['e5'], 0.05, t)]


def backtest(trainingData, realData, start, end, memory, model, **kwargs):
    results = pd.DataFrame(columns=['VaR1', 'VaR5'], index=realData[start:end].index)

    timerStart = time.time()

    for date in results.index:
        print date
        f = open("log.txt", "w")
        f.write('start: ' + str(start) + '\n')
        f.write('end: ' + str(end) + '\n')
        f.write('now: ' + str(date.strftime('%Y%m%d')) + '\n')
        f.close()
        dateMemory = date - datetime.timedelta(days=memory)
        modelSim1p, modelSim5p = model(trainingData[dateMemory:date], **kwargs)
        results.loc[date] = [modelSim1p, modelSim5p]

    duration = (time.time() - timerStart) / len(results.index)

    tests = formalTests(results, realData)
    tests.append(duration)
    backtestRapport = pd.DataFrame([t for t in tests]
                                   , index=[
            'unconditional coverage 1%',
            'unconditional coverage 5%',
            'proportion of failures 1%',
            'proportion of failures 5%',
            'time until first failure 1%',
            'time until first failure 5%',
            'Christoffersen ift 1%',
            'Christoffersen ift 5%',
            'mixed Kupiec test 1%',
            'mixed Kupiec test 5%',
            'comp.duration per day'])

    return backtestRapport


if __name__ == "__main__":
    df = pd.read_csv('data/minutedata4.csv', sep=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = np.log(df).diff().dropna()
    print "data loaded", time.time() - t0
    backtest_output = backtest(trainingData=df, realData=realizedDaily(), start='20130301', end='20150806', memory=50,
                                   model=estimateAndBootstrap, H=15, iter=10000, sparse_method=True)

    file = open("basemodel" + time.strftime("%Y%m%d", time.gmtime()) + ".txt", "w")
    file.write("After cleansing data, second run at backtesting the fully specified version\n \n")
    for a, b in zip(backtest_output.index, backtest_output.values):
        file.write('{:30}'.format(a) + ",\t" + str(b[0]) + "\n")
    file.close()
