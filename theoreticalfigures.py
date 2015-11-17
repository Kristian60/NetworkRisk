from __future__ import  division

__author__ = 'Thomas'
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import matplotlib.ticker as plticker
from scipy import stats
import matplotlib

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

if __name__  == "__main__":
    BGallo()
    exit()
    FormalTests()
    VaR()