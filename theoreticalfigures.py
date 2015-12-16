from __future__ import  division

__author__ = 'Thomas'
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import matplotlib.ticker as plticker
from scipy import stats
from scipy.stats import expon
import matplotlib
from statsmodels.tsa.stattools import acf
import random



pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)


def VaR():
    a = np.random.normal(0,0.2,1000000)
    q5 = np.percentile(a,0.05)
    q1 = np.percentile(a,0.01)

    plt.figure(figsize=(12,8))
    seaborn.kdeplot(a,label='Portfolio Returns')
    plt.vlines(q5,0,2.5,linestyles='--',label='$5 \%$ VaR')
    plt.vlines(q1,0,2.5,linestyles='--',label='$1 \%$ VaR',color = 'red')
    plt.ylim(0,2.1)
    plt.xlim(-1,1)
    plt.xticks([-1,0,1],[''])
    plt.yticks([0,1],[''])
    plt.annotate('$\mu$',xy=(0,0),ha='center',va='center')
    plt.legend()
    plt.savefig('Graphs/VAR.pdf')
    plt.show()
def FormalTests():
    T = 1000
    p = 0.05
    xV = range(40,61)
    print [x for x in xV]
    frt = [(x-p*T)/(np.sqrt(p*(1-p)*T)) for x in xV]
    pof = [-2*np.log((np.power(1-p,T-x)*np.power(p,x))/(np.power(1-x/T,T-x)*np.power(x/T,x))) for x in xV]

    plt.plot(xV,pof)
    plt.show()
    exit()
def BGallo():

    np.random.seed(10)
    matplotlib.rcParams['legend.fontsize'] = '18'

    def BG_algo(nbrhd, n, d, y, obs):
        nbrhd = nbrhd[nbrhd.index != n]
        tmd_mean = stats.trim_mean(nbrhd, d)
        std = np.std(nbrhd)
        obs_dif = abs(obs-tmd_mean)
        acc = 3*std+y
        return (obs_dif <= acc), tmd_mean, std

    #df = pd.read_csv('GE-15jan2013.csv')                            # Loader data
    #df=pd.read_csv('NVO-10jul2013.csv')
    #df = df[(df.OFR != 0) & (df.BID != 0) & (df.BID <= df.OFR)]     # fjerner 0-entries, samt krav: Offer > Bid
    df = pd.DataFrame()
    df['OFR'] = np.zeros(200)

    for i in df.index:
        if i == 0:
            df.loc[i,'OFR'] = 100
        elif i == 100:
            df.loc[i,'OFR'] = df.loc[i-1,'OFR'] + np.random.normal() + 10
            #df.loc[i,'OFR'] = df.loc[i-1,'OFR'] + np.random.normal()
        elif i == 101:
            df.loc[i,'OFR'] = df.loc[i-2,'OFR'] + np.random.normal()
        else:
            df.loc[i,'OFR'] = df.loc[i-1,'OFR'] + np.random.normal()



    df['BID'] = df['OFR'] - 0.01
    df['BnG'] = pd.Series(np.ones_like(df.OFR))                     # ny attribut BnG, som udgangspunkt = 1
    df['Nbhd_mean'] = pd.Series(np.zeros_like(df.OFR))
    df['Nbhd_std'] = pd.Series(np.zeros_like(df.OFR))

    df.index = range(len(df.index))
    print df.index[-1:]
    df = df[0:1000]

    print df.describe()

    k = 20      # neighborhood
    y = np.percentile(abs(df['OFR'].pct_change()),99)    # Granulity parameter
    d = 0.10    # Trimming factor

    print y
    for n in df.index:
        if n <= k/2:
            OFR_nbrhd = df.OFR[:k]
            BID_nbrhd = df.BID[:k]

        elif n >= (len(df)-(k/2)):
            OFR_nbrhd = df.OFR[-k:]
            BID_nbrhd = df.BID[-k:]
        else:
            OFR_nbrhd = df.OFR[int(n-(k/2)):int(n+(k/2))]
            BID_nbrhd = df.BID[int(n-(k/2)):int(n+(k/2))]

        df.BnG[n], df.Nbhd_mean[n], df.Nbhd_std[n] = BG_algo(OFR_nbrhd, n, d, y, df.OFR[n])

    acc_set = df[(df.BnG == 1)]
    fail_set = df[(df.BnG != 1)]
    df['acc_min'] = df.Nbhd_mean-3*df.Nbhd_std-y
    df['acc_max'] = df.Nbhd_mean+3*df.Nbhd_std+y


    fig = plt.figure(num=1, figsize=(20, 10), facecolor='grey')
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.plot(df.Nbhd_mean, c='b', alpha=0.5,label='Neighborhood Mean')
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=50.0))
    ax.fill_between(df.index, df.acc_min.values, df.acc_max.values, alpha=0.5)
    ax.scatter(acc_set.index,acc_set.OFR, marker='o', c='black',label='Valid Observations',s=60)
    ax.scatter(fail_set.index,fail_set.OFR, marker='o', c='r',label='Invalid Observations',s=60)
    ax.set_yticklabels(range(95,145,5),[str(j) for j in range(95,145,5)],size = 20)
    ax.set_xticklabels(range(-50,250,50),[str(j) for j in range(-50,250,50)],size = 20)
    ax.set_xlim(0,len(df))


    plt.legend(loc=2)
    plt.savefig('graphs/BrGalloAlgo.pdf',bbox_inches='tight')
    plt.show()

def DecayBoot():

    days = 10
    a = [expon.pdf(j/1000+1,scale=1.5)+.02 for j in np.arange(0,390*days,1)]
    b = []
    for i in np.linspace(1,3,days):
        b.extend([expon.pdf(i) for _ in range(390)])
    fig,axs = plt.subplots(1,1)
    axs.plot(b,label='Discrete Exponential Decay')
    axs.plot(a,label='Strictly Exponential Decay')
    axs.set_yticklabels([''])
    axs.set_ylabel('Relative Probability Of Selecting Observation In Bootstrap')
    axs.set_xlabel('Minutes Since Observation')
    plt.legend(loc='best')
    plt.savefig('Graphs/ExponDecay.pdf',bbox_inches='tight')
    plt.show()

def DescriptiveStatsandStylizedFacts():
    def DescStat():
        tdf = df.describe().T.dropna()
        for c in tdf.columns:
            tdf = tdf.rename(columns={c:c.title()})
        tdf = tdf.drop('Count',1)
        tdf.to_latex('DS.tex')
    def Htails():
        ##### Heavy Tails
        r = np.array(df['AAPL'].replace(np.inf,np.nan).replace(-np.inf,np.nan)).ravel()
        r = np.array([random.choice(r) for _ in range(1000000)])
        r = r[(abs(r)>=0) & (abs(r)<=0.05)]

        bw = 0.002
        seaborn.kdeplot(r,label='Return Data',bw=bw)
        ndata = np.random.normal(np.mean(r),np.std(r),int(len(r)))
        seaborn.kdeplot(ndata,label='Normally Distributed Data',bw=bw)
        plt.xlim(-0.03,0.03)
        plt.yticks([])
        plt.savefig('Graphs/HeavyTails.pdf',bbox_inches='tight')
        plt.show()
    def VolCluster():
        ##### Volatility Clustering

        r = (np.sum(df,axis=1)/len(df.columns))**2
        plt.plot_date(r.index,r,fmt='-',label='Squared Returns')
        plt.legend(loc='best')
        plt.savefig('Graphs/VolClustering.pdf',bbox_inches='tight')
    def SlowDecay():
        r = abs(np.sum(df.replace(np.inf,np.nan).replace(-np.inf,np.nan).fillna(0),axis=1)/len(df.columns))
        r = abs(df['AAPL'].replace(np.inf,np.nan).replace(-np.inf,np.nan).dropna(0))[:100]
        r1 = (np.sum(df.replace(np.inf,np.nan).replace(-np.inf,np.nan).fillna(0),axis=1)/len(df.columns))
        r1 = (df['AAPL'].replace(np.inf,np.nan).replace(-np.inf,np.nan).dropna(0))
        acf1,conf = acf(pd.DataFrame(r),alpha=0.05,nlags=20)
        acf2,conf2 = acf(pd.DataFrame(r1),alpha=0.05,nlags=20)
        plt.plot(acf1,label='ACF Absolute Returns')
        plt.plot(acf2,label='ACF Returns')
        plt.fill_between(range(len(acf1)),[i[0] for i in conf],[i[1] for i in conf],alpha=0.3)
        plt.fill_between(range(len(acf1)),[i[0] for i in conf2],[i[1] for i in conf2],alpha=0.3)
        plt.legend(loc='best')
        plt.savefig('Graphs/PACFAbsReturns.pdf',bbox_inches='tight')
    df = pd.read_csv('data/TData9313_final5.csv',index_col=0)
    df.index = pd.to_datetime(df.index)
    df = np.log(df).diff()
    #DescStat()
    SlowDecay()




if __name__  == "__main__":
    DescriptiveStatsandStylizedFacts()