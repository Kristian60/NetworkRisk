#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
import statsmodels.api as stat
import random
import scipy
import time
from statsmodels.tsa.vector_ar.var_model import ma_rep
import datetime
import math
import sys
from functools import partial
import multiprocessing as mp
from scipy.stats import expon
import theoreticalfigures
from decimal import Decimal
import pickle


if hasattr(sys, 'getwindowsversion'):
    it = 100
    import matplotlib.pyplot as plt
    import seaborn as sns

else:
    it = 10000
    import mailer
t0 = time.time()

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)


def EstimateVAR(data, H, sparse_method=False, GVD_output=True):
    """

    :param data: A numpy array of log returns
    :param H: integer, size of step ahead forecast
    :return: a dataframe of connectivity or concentration parameters
    """
    model = sm.VAR(data)
    results = model.fit(maxlags=H, ic='aic')

    SIGMA = np.cov(results.resid.T)

    if sparse_method == True:
        exit("METODEN BRUGER RESULTS.COEFS FREM FOR PARAMS")
        _nAssets = results.params.shape[1]
        _nLags = results.params.shape[0] / results.params.shape[1]

        custom_params = np.where(abs(results.params / results.stderr) > 1.96, results.params, 0)[1:].reshape(
            (_nLags, _nAssets, _nAssets))
        _ma_rep = ma_rep(custom_params, maxn=H)
    else:
        _ma_rep = results.ma_rep(maxn=H)

    GVD = np.empty_like(SIGMA)

    if GVD_output:
        r, c = GVD.shape
        for i in range(r):
            for j in range(c):
                GVD[i, j] = 1 / np.sqrt(SIGMA[j, j]) * sum([_ma_rep[h, i].dot(SIGMA[j]) ** 2 for h in range(H)]) / sum(
                    [_ma_rep[h, i, :].dot(SIGMA).dot(_ma_rep[h, i, :]) for h in range(H)])
            GVD[i] /= GVD[i].sum()

    return pd.DataFrame(GVD), SIGMA, _ma_rep, results.resid


def BootstrapMult(resid, marep, nIterations, dummy=False, decay=True, report_traces=False, report_individual_traces = False, report_marginalDist=False,
                  graph_distribution=False):
    '''

    Ikke færdiggjort.
    Funktionene skal replikere "nIterations" perioders afkast af "periods" længde ved at bootstrappe shockvektorer fra
    "resid"

    :param resid:
    :param marep:
    :param nIterations:
    :return:
    '''
    # Number of periods to simulate, and length of the response to shocks
    periods = int(60 * 6.5 + 1)  # en dag i minutter
    responseLength = len(marep)
    nAssets = len(marep[0])

    dailyReturns = []
    dailyTraces = []
    marginalReturn = []
    residNp = resid.values

    impulseResponseSystem = marep[::-1]  # Invert impulse responses to fit DataFrame

    simR = np.empty((periods, nAssets))
    simV = np.ones((periods + 1, nAssets))

    if decay:
        shockM = bootstrapExpDecay(resid, nIterations)
    else:
        shockM = np.array([[random.choice(residNp) for x in range(len(simR) + 15)] for nn in range(nIterations)])

    for i in range(nIterations):
        simReturns = simR.copy()
        simValues = simV.copy()
        shockMatrix = shockM[i]

        if dummy == True:
            pseudoReturn = np.product(np.sum(shockMatrix[15:] + 1, axis=1) / 11)
            dailyReturns.append(pseudoReturn)
        else:
            for t in range(len(simReturns)):
                simReturns[t] = sum([impulseResponseSystem[h].dot(shockMatrix[t + h - responseLength + 1]) for h in
                                     range(responseLength)])
                simValues[t + 1] *= simValues[t] * (simReturns[t] + 1)

            if report_traces:
                dailyTraces.append(simValues.sum(axis=1) / 17)

            if report_individual_traces:
                dailyTraces.append(simValues)

            if report_marginalDist:
                marginalReturn.append(simValues[-1, :])

            dailyReturns.append(simValues[-1, :].sum() / simValues.shape[1])

    if report_traces:
        return np.array(dailyTraces)

    if report_individual_traces:
        output = open('data.pkl', 'wb')
        pickle.dump(dailyTraces, output)
        output.close()
    if report_marginalDist:
        theoreticalfigures.graph_marginalDist(marginalReturn, resid.columns)
    if graph_distribution:
        theoreticalfigures.graph_pdf(dailyReturns)

    return dailyReturns


def bootstrapExpDecay(data, nIterations):
    d1 = data.index[-1]
    d1 = datetime.datetime(d1.year, d1.month, d1.day)

    shapeval = 100
    daysSince = [(d1 - j).days + 1 for j in data.index]

    probDist = [1 - expon.cdf(j, scale=shapeval) for j in np.unique(daysSince)]
    probDist /= np.sum(probDist)
    probDist = np.cumsum(sorted(probDist, reverse=True))

    minsPerDay = data.resample("d", how="count").values[:, 0]
    utilizedLags = int(391 - minsPerDay[0])
    bootstrapLength = 391 + utilizedLags
    data = np.insert(data.values, 0, np.zeros_like(data.ix[:utilizedLags, :]), axis=0)
    data = data.reshape((len(data) / 391, 391, data.shape[1]))
    uninumbers = np.random.uniform(size=(bootstrapLength, nIterations))

    a = np.array([np.digitize(uninumbers[_], probDist) for _ in range(bootstrapLength)])
    b = np.array([[random.choice(data[-i, :, :]) for i in a[:, q]] for q in range(nIterations)])

    return b


def VarFormalTest(results, alpha, data, name=None):
    data = _genDailyReturns(data)
    results = results - 1
    alpha = float(alpha)

    def unconditionalCoverage(events, t):
        '''
        Simple binomial test of event frequency, Jorion(2001)
        :param events: a pandas series of True/False events of events occuring
        :param p: true likelihood of event
        :param t: length of test
        :return: the p-value of the test
        '''
        x = sum(events)
        return (x - alpha * t) / np.sqrt(alpha * (1 - alpha) * t), \
               scipy.stats.norm.cdf((x - alpha * t) / np.sqrt(alpha * (1 - alpha) * t))

    def pofTest(events, t, raw_output=False):
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
                (np.power(1 - alpha, t - x) * np.power(alpha, x)) / (np.power(1 - (x / t), t - x) * np.power(x / t, x)))
        else:

            opt_breaks = Decimal(alpha) ** Decimal(x)
            opt_nobreaks = Decimal(1 - alpha) ** Decimal((t - x))
            model_breaks = Decimal(x / t) ** Decimal(x)
            model_nobreaks = Decimal(1 - (x / t)) ** Decimal(t - x)

            num = (opt_breaks * opt_nobreaks)
            den = (model_breaks * model_nobreaks)

            aux = num / den
            aux = float(aux)

            stat = -2 * np.log(aux)

            return stat, scipy.stats.chi2.cdf(stat, df=1)

    def tuffTest(events, raw_output=False):
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
                return (
                    v, -2 * np.log((alpha * np.power((1 - alpha), v - 1)) / ((1 / v) * np.power(1 - (1 / v), v - 1))))
            else:
                return -2 * np.log(
                    (p * np.power((1 - p), v - 1)) / ((1 / v) * np.power(1 - (1 / v), v - 1))), scipy.stats.chi2.cdf(
                    -2 * np.log((p * np.power((1 - p), v - 1)) / ((1 / v) * np.power(1 - (1 / v), v - 1))), df=1)

    def christoffersenIFT(events, t):
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

        if n00 + n10 == 0:
            n1 = 1
        else:
            n1 = Decimal(1 - pi) ** Decimal(n00 + n10)

        if n01 + n11 == 0:
            n2 = 1
        else:
            n2 = Decimal(pi) ** Decimal(n01 + n11)

        num = n1 * n2

        if n00 == 0:
            d1 = 1
        else:
            d1 = Decimal(1 - pi0) ** Decimal(n00)

        if n01 == 0:
            d2 = 1
        else:
            d2 = Decimal(pi0) ** Decimal(n01)

        if n10 == 0:
            d3 = 1
        else:
            d3 = Decimal(1 - pi1) ** Decimal(n10)
        if n11 == 0:
            d4 = 1
        else:
            d4 = Decimal(pi1) ** Decimal(n11)

        dem = d1 * d2 * d3 * d4

        LRind = -2 * np.log(float(num / dem))

        return LRind, scipy.stats.chi2.cdf(LRind, df=1)

    def mixedKupiecTest(events, t):

        nEvents = sum(events)
        LRind = 0
        for e in range(nEvents):
            n, LRtuff = tuffTest(events, raw_output=True)
            LRind += LRtuff
            events = events[n:]

        return LRind / nEvents, scipy.stats.chi2.cdf(LRind, nEvents)

    break_data = pd.concat([results, data], axis=1, keys=["results", "data"]).dropna()
    break_data['breaks'] = break_data['data'] < break_data['results']
    T = float(len(break_data))
    events = break_data['breaks']

    tests = pd.DataFrame(index=["Unconditional coverage", "POF test", "Kupiec test", "Christoffersen test"],
                         columns=['stat', 'pval', 'accepted','type', 'interpret p'])

    uc_s, uc_p = unconditionalCoverage(events, T)
    pf_s, pf_p = pofTest(events, T)
    ku_s, ku_p = mixedKupiecTest(events, T)
    ch_s, ch_p = christoffersenIFT(events, T)

    tests.loc["Unconditional coverage"] = [uc_s, uc_p, 0.05 < uc_p < 0.95, "two sided","0.05 < p < 0.95"]
    tests.loc["POF test"] = [pf_s, pf_p, pf_p < 0.95, "one sided","       p < 0.95"]
    tests.loc["Kupiec test"] = [ku_s, ku_p, ku_p < 0.95, "one sided","       p < 0.95"]
    tests.loc["Christoffersen test"] = [ch_s, ch_p, ch_p < 0.95, "one sided","       p < 0.95"]

    if name:
        tests.to_csv('FormalTests_'+ name +'.csv')
    else:
        tests.to_csv('FormalTests_'+datetime.datetime.now().strftime("%Y%m%d")+'.csv')
    return

def ESFormalTest(es, var, alpha, data, name=None):
    data = _genDailyReturns(data)
    es = es - 1
    var = var - 1

    break_data = pd.concat([es,var, data], axis=1, keys=["es","var", "data"]).dropna()
    break_data['breaks'] = break_data['data'] < break_data['var']

    break_data = break_data[break_data['breaks']]


    break_data['stat'] = break_data['data']/break_data['es']

    print break_data['stat'].mean()
    return

def BerkowitzTest(bD):


    bD = ((bD - 0.5) / 100).apply(scipy.stats.norm.ppf).dropna()
    return stat.stats.stattools.jarque_bera(bD)

def generateBerkowizData(returnSeries,realized):

    returnSeries = returnSeries
    result = []
    max_breaks = 0
    min_breaks = 0
    for day in returnSeries.index:
        buckets = [np.percentile(returnSeries.loc[day],x) for x in range(101)]
        perc = sum(realized.loc[day] > buckets)

        if perc == 101:
            print day
            max_breaks += 1
            perc = 100

        if perc == 0:
            min_breaks += 1
            perc = 1

        result.append(perc)

    print "max_breaks", max_breaks
    print "min_breaks", min_breaks
    pd.DataFrame(result).to_csv('berkowiz_bench2.csv')
    exit()



def benchmarkModel(data, bootstrapPoolDays=500, saveSimulations=False):
    data = _genDailyReturns(data)
    firstDay = data.index[bootstrapPoolDays]
    relevantDaysSet = data[firstDay:]

    out_df = pd.DataFrame()
    output = np.array([])
    for date in relevantDaysSet.index:
        real_return = data.loc[date]

        bootstrapPool = data[:date]
        activeAssets = bootstrapPool[-50:].values
        bootstrapPool = bootstrapPool[-500:]

        draw = [random.choice(bootstrapPool.values) for x in range(10000)]

        if saveSimulations:
            returnSeries = pd.DataFrame([draw]).apply(np.round,decimals=5)
            returnSeries.index = [date]
            returnSeries.to_csv('rSeries_benchmark.csv',mode="a",header=False)

#        var1 = np.percentile(draw, 1)
#        var5 = np.percentile(draw, 5)
#        es1 = np.mean(np.extract(draw < var1, draw))
#        es5 = np.mean(np.extract(draw < var5, draw))

#        output = np.append(output, [date, var1, var5, es1, es5, 1 + real_return[activeAssets].mean()])

#    out_df = pd.DataFrame(output.reshape((len(output) / 6, 6)),
#                          columns=['Date', 'Var1', 'Var5', 'ES1', 'ES5', 'Real Values'])
    print "Done"
    exit()
    out_df.to_csv('benchmark_model.csv')

    return


def backtestthread(trainingData, realData, start, end, memory, model):
    results = pd.DataFrame(columns=['VaR1', 'VaR5', 'ES1', 'ES5'], index=realData[start:end].index)

    timerStart = time.time()
    func = partial(btestthread, start, end, memory, model, trainingData, results)
    for nrthreads in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100]:
        t = results.iloc[:nrthreads, :].index
        pool = mp.Pool(nrthreads)
        timerStart = time.time()
        output = pool.map(func, t)
        pool.close()
        pool.join()
        print (time.time() - timerStart) / nrthreads

    duration = (time.time() - timerStart) / len(results.index)

    return


def _genLogReturn(data):
    data = np.log(data).diff().dropna(how='all')
    return data


def _genDailyReturns(data):
    data = data.resample('d', how='last').dropna(how='all')
    data = np.log(data).diff().dropna(how='all')
    data = data.mean(axis=1, numeric_only=True)
    return data


def NetworkModel(trainingData, start, end, memory, saveSimulations=False):

    trainingData = _genLogReturn(trainingData)
    daily = trainingData.resample('d', how='last').dropna(how='all')
    results = pd.DataFrame(columns=['VaR1', 'VaR5', 'ES1', 'ES5'], index=daily[start:end].index)
    timerStart = time.time()

    accReturnSeries = pd.DataFrame([])

    for date in results.index:
        dStart = time.time()
        print date
        dateMemory = date - datetime.timedelta(days=memory)
        df = trainingData[dateMemory:date].dropna(axis=1, how='any')
        df = df.dropna(axis=1, how='any')
        con, sigma, marep, resid = EstimateVAR(df, H=15)
        returnSeries = BootstrapMult(resid, marep, it)

        if saveSimulations:
            returnSeries = pd.DataFrame([returnSeries]).apply(np.round,decimals=5)
            returnSeries.index = [date]
            returnSeries.to_csv('rSeries_nwrk.csv',mode="a",header=False)

        var1 = np.percentile(returnSeries, 1)
        var5 = np.percentile(returnSeries, 5)
        es1 = np.mean(np.extract(returnSeries < var1, returnSeries))
        es5 = np.mean(np.extract(returnSeries < var5, returnSeries))

        results.loc[date] = [var1, var5, es1, es5]
        results.to_csv('dailyResults_' + time.strftime("%Y%m%d", time.gmtime()) + ".csv")
        print time.time() - dStart


    duration = (time.time() - timerStart) / len(results.index)
    return


if __name__ == "__main__":
    #print BerkowitzTest(pd.read_csv('berkowiz_bench2.csv',index_col=0,header=None,names=['perc','obs']))
    #exit()
    try:
        df = pd.read_csv('data/TData9313_final6.csv', sep=",", index_col=0)
        print "data loaded", time.time() - t0
        df.index = pd.to_datetime(df.index)

#    res = pd.read_csv('resultstest.csv', index_col='Date')
#    res.index = pd.to_datetime(res.index, format='%d-%m-%Y')

        NetworkModel(df,'19941227','20150101',100,saveSimulations=True)
    #benchmarkModel(df,saveSimulations=True)
    #exit()

    #retS = pd.read_csv('rSeries_benchmark.csv', index_col=0)
    #retS.index = pd.to_datetime(retS.index, format='%Y-%m-%d')

    #generateBerkowizData(retS,_genDailyReturns(df))
    #ESFormalTest(es=res['bnch_ES5'], var=res['bnch_VaR5'], alpha=0.05, data=df, name='bnchES5p')


    #NetworkModel(trainingData=df, start='20121227', end='20150101', memory=100, saveSimulations=True)

        mailer.send('dailyResults_' + time.strftime("%Y%m%d", time.gmtime()) + ".csv", 'holden750@gmail.com',
                    'Ireren er færdig')
        mailer.send('dailyResults_' + time.strftime("%Y%m%d", time.gmtime()) + ".csv", 'thorup.dk@gmail.com',
                    'Ireren er færdig')

    except:
        mailer.send('dailyResults_' + time.strftime("%Y%m%d", time.gmtime()) + ".csv", 'holden750@gmail.com',
                    'Ireren har fejlet')
        mailer.send('dailyResults_' + time.strftime("%Y%m%d", time.gmtime()) + ".csv", 'thorup.dk@gmail.com',
                    'Ireren har fejlet')
