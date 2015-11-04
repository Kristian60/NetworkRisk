from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from Connectedness import EstimateVAR
import Connectedness
import pandas as pd
import random
import seaborn
import datetime
import statsmodels.tsa.api as sm
from statsmodels.tsa.vector_ar.var_model import ma_rep
import scipy.stats

def MakeMinute():
    df = pd.read_csv('data/full_data.csv',iterator=True,chunksize=10000)
    for enr, data in enumerate(df):
        print enr
        data['seconds'] = [j.split(':')[-1] for j in (data['TIME'])]
        data = data[data['seconds']=='00']
        data = data.set_index(pd.to_datetime(data['DATE'] + ' ' + data['TIME']))
        data = data[[j for j in data.columns if j not in ['TIME','DATE','seconds']]]
        #data = data.asfreq('1Min')
        if enr ==0:
            data[:0].to_csv('minutedata2.csv',mode='w',index=True,header=True)
        data.to_csv('minutedata2.csv',mode='a',index=True,header=False)
def Old(data):
    test = []

    for i in range(10):
        t0 = datetime.datetime.now()
        bStrapData = pd.DataFrame(np.array([data.ix[random.randint(0,len(data)-1),:] for x in range(len(data))]),
                                 index=data.index,columns=data.columns)

        bStrapData.plot()
        data.plot()
        plt.show()
        exit()


        print datetime.datetime.now()-t0
        print data
        print bStrapData
        gvd, sigma, ma_rep, resid = EstimateVAR(bStrapData,15)
        print pd.DataFrame(ma_rep[1])
        exit()
        print ma_rep[1][0,0]
        test.append(ma_rep[1][0,0])
    plt.hist(test)
def EstimateVARTest(data, H, sparse_method=False):
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
            #GVD[i, j] = 1 / np.sqrt(SIGMA[i, i]) * sum([_ma_rep[h, i].dot(SIGMA[j]) ** 2 for h in range(H)]) / sum([_ma_rep[h, i, :].dot(SIGMA).dot(_ma_rep[h, i, :]) for h in range(H)])
            GVD[i, j] = sum([_ma_rep[h, i].dot(SIGMA[j]) ** 2 for h in range(H)]) / sum([_ma_rep[h, i, :].dot(SIGMA).dot(_ma_rep[h, i, :]) for h in range(H)])
        #GVD[i] /= GVD[i].sum()


    print pd.DataFrame(SIGMA)*10000000
    print pd.DataFrame(GVD)*10000000
    print pd.DataFrame(SIGMA)-pd.DataFrame(GVD)

    return pd.DataFrame(GVD), SIGMA, _ma_rep, results.resid



def VarSimul(data,H):
    model = sm.VAR(data)
    results = model.fit(H)
    VARcoeff = results.params[1:]
    VARcoeff = np.array(VARcoeff).reshape(len(VARcoeff)/len(data.columns),len(data.columns),len(data.columns))
    VARstd = results.stderr[1:]
    VARstd = np.array(VARstd).reshape(len(VARstd)/len(data.columns),len(data.columns),len(data.columns))
    test = []
    for i in range(1000):
        VarSim = np.zeros((len(VARcoeff)/len(data.columns),len(data.columns),len(data.columns)))
        for j in range(VarSim.shape[0]):
            for k in range(VarSim.shape[1]):
                for l in range(VarSim.shape[2]):
                    VarSim[j][k,l] = np.random.normal(VARcoeff[j][k,l],VARstd[j][k,l])
        marep = ma_rep(VarSim,10)
        test.append(marep[1][0,0])
    print np.std(test)
    seaborn.distplot(test,norm_hist=True)
    plt.show()
def MetropolisHastingMCMC(data,H):
    model = sm.VAR(data)
    results = model.fit(H)
    VARcoeff = results.params[1:]
    VARcoeff = np.array(VARcoeff).reshape(len(VARcoeff)/len(data.columns),len(data.columns),len(data.columns))
    VARstd = results.stderr[1:]
    VARstd = np.array(VARstd).reshape(len(VARstd)/len(data.columns),len(data.columns),len(data.columns))
    mean = VARcoeff[1][1,1]
    std = VARstd[1][1,1]
    #mean = 1000
    #std = 100
    N = 200000
    s = 10
    r = np.random.normal(mean,std)
    #p = np.random.normal(mean,std)
    p = scipy.stats.norm.pdf(r,mean,std)+ 1
    samples=[]
    #plt.ion()
    #plt.show()
    for i in range(N):
        if i%10000==0:
            print i
        rn = r + np.random.normal()
        pn = scipy.stats.norm.pdf(rn,mean,std)
        if pn >= p:
            p = pn
            r = rn
        else:
            u = np.random.rand()
            if u < pn/p:
                p = pn
                r = rn
        if i % s == 0:
            samples.append(r)
            #plt.plot(samples)
            #plt.draw()



    samples = samples[int(N/200):]
    normdata = np.random.normal(mean,std,len(samples))
    seaborn.distplot(samples,norm_hist=True,label='MCMC')
    seaborn.distplot(normdata,norm_hist=True,label='normdata')
    plt.vlines(mean,0,7.5)
    plt.legend()
    plt.show()
def WildBootstrap(data,H):
    gvd,sigma,vma,resid = EstimateVAR(data,H)
    nAssets = len(vma[0])
    periods = int(60 * 6.5)
    responseLength = len(vma)
    simReturns = np.zeros((periods, nAssets))
    simValues = np.ones((periods + 1, nAssets))
    fig1,axs = plt.subplots(2,2,figsize=(20,8),facecolor='grey',edgecolor = 'black')
    fig1.subplots_adjust(hspace = .5, wspace=.5)
    axs = axs.ravel()
    methods = ['regular','std','mammen','rademacher']
    ymin = 0
    ymax = 0
    for i,method in enumerate(methods):
        random.seed(0)
        np.random.seed(0)
        if method == 'regular':
            shockMatrix = np.array([random.choice(resid.values) for x in range(len(simReturns) + 15)])
        elif method == 'std':
            shockMatrix = np.array([random.choice(resid.values)*np.random.normal() for x in range(len(simReturns) + 15)])
        elif method == 'mammen':
            shockMatrix = np.array([random.choice(resid.values)*(-(np.sqrt(5)-1)/2) if np.random.rand() < (np.sqrt(5)-1)/(2*np.sqrt(5)) else random.choice(resid.values)*((np.sqrt(5)+1)/2) for x in range(len(simReturns) + 15)])
        elif method == 'rademacher':
            shockMatrix = np.array([random.choice(resid.values) if np.random.rand()<0.5 else random.choice(resid.values)*(-1) for x in range(len(simReturns) + 15)])
        impulseResponseSystem = vma[::-1]

        if np.min(shockMatrix)<ymin:
            ymin = np.min(shockMatrix)
        if np.max(shockMatrix)>ymax:
            ymax = np.max(shockMatrix)

        for t in range(len(simReturns)):
            simReturns[t] = sum([impulseResponseSystem[h].dot(shockMatrix[t + h - responseLength + 1]) for h in
                                 range(responseLength)])
            simValues[t + 1] *= simValues[t] * (simReturns[t] + 1)
        for j in range(simReturns.shape[1]):
            #axs[i].plot(simReturns[:,j])
            seaborn.kdeplot(simReturns[:,j],ax=axs[i])

        print method, '   ', np.mean(simReturns[0]), np.std(simReturns[0]), scipy.stats.skew(simReturns[0]),scipy.stats.kurtosis(simReturns[0])

    for i,method in enumerate(methods):
        #axs[i].set_ylim(ymin,ymax)
        axs[i].set_title(method)
    plt.tight_layout()
    plt.show()
def SOI(data,H):
    months = []
    sois = []
    splits = 200
    lens = int(len(data)/splits)
    #for month in range(1,7):
    ddate = datetime.datetime(2015,1,1)
    #for mm in range(splits):
    while ddate<datetime.datetime(2015,7,1):
        ddate += datetime.timedelta(1)
        #td = data[mm*lens:(mm+1)*lens]
        td = data
        #print ddate
        td.index = pd.to_datetime(td.index)
        td = td[(td.index.month==ddate.month) & (td.index.day==ddate.day) & (td.index.year==ddate.year)]
        if len(td)>0:
            gvd,sigma,marep, resid = EstimateVAR(td,15)
            soi = (len(gvd)-np.trace(np.array(gvd)))/len(gvd)
            months.append(td.index[-1])
            sois.append(soi)
            #print soi
    plt.plot_date(months,sois,fmt='-')
    plt.title('Total Spillover Index')
    plt.xlabel('Date')
    plt.ylabel('SOI')
    plt.show()


def VineCopula(dat):
    cp_dat = dat.rank() / ( len(dat) + 1 )
    ## initialize R-vine object named rv
    rv = pv.Rvine(cp_dat)
    ## sequential estimation for rv. 'structure' accepts 'r' for R-vine,
    ## 'c' for C-vine and 'd' for D-vine, 'familyset' accepts list of
    ## integers from 1 to 6, 'threads_num' accepts integer specifying number
    ## of threads using for taking mle on edges of the same vine tree
    ## simultaneously.

    rv.modeling(structure = 'r', familyset = [1,2,3,4,5,6], threads_num = 2)

    ## maximum likelihood estimation for rv. 'disp' controls the printing
    ## of ratio of progress of iterating for L-BFGS-B algorithm, 'threads_num'
    ## specifies the number of threads using for computing loglikelihood value
    ## for each edge in the same vine tree.

    rv.mle(disp=False, threads_num = 2)

    ## plot the R-vine structure for modeled object rv. All the vine trees will
    ## be plotted as default.

    rv.plot()

    ## display the result of estimation on each edge. 'ndigits' controls number
    ## of decimal digits for result.

    rv.res(ndigits = 3)

    ## testing

    rv.test()

if __name__ == "__main__":
    data = pd.read_csv('data/minutedata2.csv',index_col=0)
    data.index = pd.to_datetime(data.index)
    #data = data.asfreq('5Min').dropna()
    data = np.log(data).diff().dropna().iloc[:13000,:]
    EstimateVARTest(data,15)
    exit()
    WildBootstrap(data,15)





