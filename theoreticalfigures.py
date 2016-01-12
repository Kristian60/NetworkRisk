from __future__ import division

__author__ = 'Thomas'
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import matplotlib.ticker as plticker
import statsmodels.api as sm
from scipy.stats import expon
from statsmodels.tsa.stattools import acf
import random
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib import gridspec
import Connectedness
import datetime
import statsmodels.api as sm

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

c = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

seaborn.set(context='paper', font='Segoe UI', rc={
    'axes.facecolor': '#F0F0F0',
    'figure.facecolor': '#F0F0F0',
    'savefig.facecolor': '#F0F0F0',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'grid.color': '#DADADA',
    'ytick.color': '#66666A',
    'xtick.color': '#66666A'
})


def VaR():
    a = np.random.normal(0, 0.2, 1000000)
    q5 = np.percentile(a, 0.05)
    q1 = np.percentile(a, 0.01)

    plt.figure(figsize=(12, 8))
    seaborn.kdeplot(a, label='Portfolio Returns')
    plt.vlines(q5, 0, 2.5, linestyles='--', label='$5 \%$ VaR')
    plt.vlines(q1, 0, 2.5, linestyles='--', label='$1 \%$ VaR', color='red')
    plt.ylim(0, 2.1)
    plt.xlim(-1, 1)
    plt.xticks([-1, 0, 1], [''])
    plt.yticks([0, 1], [''])
    plt.annotate('$\mu$', xy=(0, 0), ha='center', va='center')
    plt.legend()
    plt.savefig('Graphs/VAR.pdf')
    plt.show()


def FormalTests():
    T = 1000
    p = 0.05
    xV = range(40, 61)
    print [x for x in xV]
    frt = [(x - p * T) / (np.sqrt(p * (1 - p) * T)) for x in xV]
    pof = [-2 * np.log((np.power(1 - p, T - x) * np.power(p, x)) / (np.power(1 - x / T, T - x) * np.power(x / T, x)))
           for x in xV]

    plt.plot(xV, pof)
    plt.show()
    exit()


def BGallo():
    np.random.seed(10)

    def BG_algo(nbrhd, n, d, y, obs):
        nbrhd = nbrhd[nbrhd.index != n]
        tmd_mean = stats.trim_mean(nbrhd, d)
        std = np.std(nbrhd)
        obs_dif = abs(obs - tmd_mean)
        acc = 3 * std + y
        return (obs_dif <= acc), tmd_mean, std

    # df = pd.read_csv('GE-15jan2013.csv')                            # Loader data
    # df=pd.read_csv('NVO-10jul2013.csv')
    # df = df[(df.OFR != 0) & (df.BID != 0) & (df.BID <= df.OFR)]     # fjerner 0-entries, samt krav: Offer > Bid
    df = pd.DataFrame()
    df['OFR'] = np.zeros(200)

    for i in df.index:
        if i == 0:
            df.loc[i, 'OFR'] = 100
        elif i == 75:
            df.loc[i, 'OFR'] = df.loc[i - 1, 'OFR'] + np.random.normal() + 6
            # df.loc[i,'OFR'] = df.loc[i-1,'OFR'] + np.random.normal()
        elif i == 76:
            df.loc[i, 'OFR'] = df.loc[i - 2, 'OFR'] + np.random.normal()
        else:
            df.loc[i, 'OFR'] = df.loc[i - 1, 'OFR'] + np.random.normal()

    df['BID'] = df['OFR'] - 0.01
    df['BnG'] = pd.Series(np.ones_like(df.OFR))  # ny attribut BnG, som udgangspunkt = 1
    df['Nbhd_mean'] = pd.Series(np.zeros_like(df.OFR))
    df['Nbhd_std'] = pd.Series(np.zeros_like(df.OFR))

    df.index = range(len(df.index))
    print df.index[-1:]
    df = df[0:1000]

    print df.describe()

    k = 20  # neighborhood
    y = np.percentile(abs(df['OFR'].pct_change()), 99)  # Granulity parameter
    d = 0.10  # Trimming factor

    print y
    for n in df.index:
        if n <= k / 2:
            OFR_nbrhd = df.OFR[:k]
            BID_nbrhd = df.BID[:k]

        elif n >= (len(df) - (k / 2)):
            OFR_nbrhd = df.OFR[-k:]
            BID_nbrhd = df.BID[-k:]
        else:
            OFR_nbrhd = df.OFR[int(n - (k / 2)):int(n + (k / 2))]
            BID_nbrhd = df.BID[int(n - (k / 2)):int(n + (k / 2))]

        df.BnG[n], df.Nbhd_mean[n], df.Nbhd_std[n] = BG_algo(OFR_nbrhd, n, d, y, df.OFR[n])

    acc_set = df[(df.BnG == 1)]
    fail_set = df[(df.BnG != 1)]
    df['acc_min'] = df.Nbhd_mean - 3 * df.Nbhd_std - y
    df['acc_max'] = df.Nbhd_mean + 3 * df.Nbhd_std + y

    fig = plt.figure(num=1)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.plot(df.Nbhd_mean, alpha=0.4, label='Neighborhood Mean', color=c[1])
    # ax.xaxis.set_major_locator(plticker.MultipleLocator(base=50.0))
    ax.fill_between(df.index, df.acc_min.values, df.acc_max.values, alpha=0.2, color=c[2])
    ax.scatter(acc_set.index, acc_set.OFR, marker='o', color='#000000', alpha=0.6, label='Valid Observations', s=5)
    ax.scatter(fail_set.index, fail_set.OFR, marker='o', color=c[3], label='Invalid Observations', s=5)
    # ax.set_yticklabels(range(95,145,5),[str(j) for j in range(95,145,5)],size = 2)
    # ax.set_xticklabels(range(-50,250,50),[str(j) for j in range(-50,250,50)],size = 2)
    ax.set_xlim(1, len(df) - 5)
    ax.set_ylim(96, 129)

    plt.legend(loc=2)
    plt.savefig('BrGalloAlgo.pdf')
    # plt.show()


def DecayBoot():
    days = 100
    b = []

    for i in range(1, days):
        b.extend([1-expon.cdf(i,scale=100) for _ in range(390)])
    a = [1-expon.cdf(j, scale=100) + 0.003 for j in np.linspace(1, days, len(b))]
    fig, ax = plt.subplots(1, 2,figsize=(18,6))

    for nr,axs in enumerate(ax):
        if nr == 0:
            axs.plot(b, label='Discrete Exponential Decay', color=c[1],lw=1)
            axs.plot(a, label='Strictly Exponential Decay', color=c[2],lw=1)
            axs.set_ylabel('Relative probability of selecting observation')
            axs.set_xlabel('Days since observation')
            axs.set_ylim(-0.05,1.05)
            axs.set_xlim(-2500,len(b)+2500)
            axs.set_xticks(range(0,len(b)+3900,3900))
            axs.set_xticklabels(range(0,110,10))
        else:
            axs.plot(b, label='Discrete Exponential Decay', color=c[1])
            axs.plot(a, label='Strictly Exponential Decay', color=c[2])
            axs.set_xlim(-250,3900+250)
            axs.set_ylim(0.87,1.01)
            axs.set_xticks(range(0,3900+390,390))
            axs.set_xticklabels(range(0,11,1))

    plt.legend(loc='best')
    plt.savefig('Graphs/ExponDecay.pdf',bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def DescriptiveStatsandStylizedFacts():
    def DescStat():
        tdf = df.describe().T.dropna()
        for c in tdf.columns:
            tdf = tdf.rename(columns={c: c.title()})
        tdf = tdf.drop('Count', 1)
        tdf.to_latex('DS.tex')

    def Htails():
        ##### Heavy Tails
        r = np.array(df['AAPL'][-200000:].replace(np.inf, np.nan).replace(-np.inf, np.nan)).ravel()
        # r = np.array([random.choice(r) for _ in range(1000000)])
        # r = r[(abs(r)>=0) & (abs(r)<=0.05)]

        bw = 0.002
        seaborn.distplot(r, bins=801 * 3, kde=False, label='Return Data', norm_hist=True, color=c[1])
        ndata = np.random.normal(np.mean(r), np.std(r), int(len(r)))
        seaborn.kdeplot(ndata, label='Normal Distribution', linestyle="-", color=c[0])
        plt.xlim(-0.0031, 0.0031)
        plt.ylim(-30, 1120)
        plt.yticks([0, 200, 400, 600, 800, 1000], ['', '', '', '', '', ''])
        plt.savefig('HeavyTails.pdf')

    #        plt.show()
    def VolCluster():
        ##### Volatility Clustering

        fig1 = plt.figure(num=None, figsize=(7, 4), dpi=300, facecolor='0.95')
        ax1 = fig1.add_subplot(1, 1, 1)

        r = (np.sum(df, axis=1) / len(df.columns)) ** 2
        ax1.plot_date(r.index, r, fmt='-', label='Squared Returns', color=c[1])
        ax1.set_ylim(-0.0001, 0.0031)
        plt.legend(loc='best')
        # plt.show()
        plt.savefig('VolClustering.pdf')

    def SlowDecay():
        r = abs(np.sum(df.replace(np.inf, np.nan).replace(-np.inf, np.nan).fillna(0), axis=1) / len(df.columns))
        r = abs(df['AAPL'].replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna(0))[:1000000]
        r1 = (np.sum(df.replace(np.inf, np.nan).replace(-np.inf, np.nan).fillna(0), axis=1) / len(df.columns))
        r1 = (df['AAPL'].replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna(0))
        acf1, conf = acf(pd.DataFrame(r), alpha=0.05, nlags=20)
        acf2, conf2 = acf(pd.DataFrame(r1), alpha=0.05, nlags=20)
        plt.plot(acf1, label='ACF Absolute Returns')
        plt.plot(acf2, label='ACF Returns')
        plt.fill_between(range(len(acf1)), [i[0] for i in conf], [i[1] for i in conf], alpha=0.3)
        plt.fill_between(range(len(acf1)), [i[0] for i in conf2], [i[1] for i in conf2], alpha=0.3)
        plt.legend(loc='best')
        plt.savefig('Graphs/PACFAbsReturns.pdf', bbox_inches='tight')

    def tracesGraph(df):
        GVD, SIGMA, _ma_rep, res = Connectedness.EstimateVAR(df, 15)
        traces = Connectedness.BootstrapMult(res, _ma_rep, 2000, report_traces=True)

        # generate some data
        x = np.arange(0, 10, 0.2)
        y = np.sin(x)

        # plot it
        fig = plt.figure(figsize=(7, 4))

        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        ax0.plot(traces.T, color=c[1], alpha=0.1, lw=0.25)

        ax1 = plt.subplot(gs[1])
        final = traces.T[-1]

        seaborn.distplot(final, ax=ax1, vertical=True, rug=True, kde=False, color=c[1], bins=200, rug_kws={'lw': 0.1})

        ax1.set_yticklabels([''])
        print ax0.get_yticklabels()
        ax0.set_yticklabels(['-5%', '-3%', '-2%', '-1%', '0%', '1%', '2%', '3%', '3%'])
        ax0.set_xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390])
        ax0.set_xticklabels(['09:30', '', '', '', '11:30', '', '', '', '13:30', '', '', '', '15:30', ''])
        ax1.set_xticklabels([''])

        ax0.vlines(0, -200, 4000, lw=0.5)
        ax1.vlines(0, -200, 4000, lw=0.5)
        ax0.hlines(1, -200, 4000, lw=0.5)
        ax0.set_ylim(0.969, 1.031)
        ax0.set_xlim(-5, 390)
        ax0.set_title('Simulated returns')
        ax1.set_ylim(0.969, 1.031)
        ax1.set_xlim(0, 37)

        ax0.set_ylabel('Return')

        plt.tight_layout(pad=2, h_pad=0, w_pad=0)
        plt.show()
        # plt.savefig('traces.pdf')

    df = pd.read_csv('data/TData9313_final5.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df['19930301':'19930501']
    df = np.log(df).diff()[1:]
    df = df.dropna(axis=1, how='all')

    # DescStat()
    # SlowDecay()
    # VolCluster()
    tracesGraph(df)


def SOIovertime():
    df = pd.read_csv("C:/Users/Thomas/Dropbox/UNI/Speciale/NetworkRisk/results/SOI_100_days.csv", index_col=0)
    df['RollMean100'] = pd.rolling_mean(df['SOI'], 100, min_periods=1)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    plt.plot_date(df.index, df['SOI'], fmt='-', label='Spillover Index', color=c[1], lw=1, alpha=0.3)
    plt.plot_date(df.index, df['RollMean100'], fmt='-', color=c[0],
                  label='100 day Rolling Mean of Spillover Index', linewidth=1, alpha=0.8)
    plt.legend(loc='best')
    plt.hlines(0, datetime.datetime(1994, 8, 1), datetime.datetime(2014, 5, 1), alpha=0.6, lw=1)
    plt.xlim(datetime.datetime(1994, 8, 1), datetime.datetime(2014, 5, 1))
    plt.ylim(-0.05, 1.05)
    plt.savefig('Graphs/SOIOvertime.pdf', bbox_inches='tight')
    plt.show()
    print df


def LLovertime():
    df = pd.read_csv("C:/Users/Thomas/Dropbox/UNI/Speciale/NetworkRisk/results/SOI_100_days.csv", index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df['RollMean100'] = pd.rolling_mean(df['LL'], 100, min_periods=1)
    plt.plot_date(df.index, df['LL'], markersize=2, fmt='-', label='Lag Length', color=c[1], alpha=0.3)
    plt.plot_date(df.index, df['RollMean100'], fmt='-', color=c[0], lw=1, alpha=0.8,
                  label='100 day Rolling Mean of Lag Length')

    ax = plt.gca()
    plt.ylim(-0.4, 16.4)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    plt.hlines(0, datetime.datetime(1994, 8, 1), datetime.datetime(2014, 5, 1), alpha=0.6, lw=1)
    plt.xlim(datetime.datetime(1994, 8, 1), datetime.datetime(2014, 5, 1))

    plt.title('Lag Length by AIC')
    plt.legend(loc='best')
    plt.savefig('Graphs/LLOvertime.pdf', bbox_inches='tight')
    plt.show()

def QQPlot():
    df = pd.read_csv('data/TData9313_final6.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = np.log(df).diff()[1:]
    df = df.dropna(axis=1, how='all')
    r = np.array(df[['AAPL']].dropna())
    norm = np.random.normal(0,np.std(r),1000000)
    x = [np.percentile(norm,j) for j in range(1,100)]
    y = [np.percentile(r,j) for j in range(1,100)]
    plt.scatter(x,y,color=c[3],s=5,alpha=0.6)
    ax = plt.gca()
    val = 0.01
    extra = 0.0025
    plt.plot([-val,val],[-val,val],color=c[1])
    plt.xlim(-val-extra,val+extra)
    plt.ylim(-val-extra,val+extra)
    plt.xlabel('Theoretical Quantile')
    plt.ylabel('Sample Quantile')
    ylabels = [str(round(j,3)*100) + ' %' for j in ax.get_yticks()]
    xlabels = [str(round(j,3)*100) + ' %' for j in ax.get_xticks()]
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)
    #plt.tight_layout()
    plt.savefig('Graphs/QQPlot.pdf',bbox_inches='tight')
    plt.show()


def resultsG1():
    df = pd.read_csv('results040116.csv', index_col=0)
    df.index = pd.to_datetime(df.index, format="%d-%m-%Y")
    df = df - 1

    for mdl in ['nwrk', 'bnch']:
        plt.fill_between(df.index, y1=df[mdl + '_VaR5'], y2=1, color=c[0], alpha=0.4, edgecolor="None", label='VaR 5%')
        plt.fill_between(df.index, y1=df[mdl + '_VaR1'], y2=df[mdl + '_VaR5'], color=c[0], alpha=0.9, edgecolor="None",
                         label='VaR 1%')

        p1 = mpatches.Patch(color=c[0], alpha=0.4, linewidth=0)
        p2 = mpatches.Patch(color=c[0], alpha=0.9, linewidth=0)

        set1 = df['realized'][df['realized'] > df[mdl + '_VaR5']]
        set2 = df['realized'][df[mdl + '_VaR1'] < df['realized']][df['realized'] < df[mdl + '_VaR5']]
        set3 = df['realized'][df['realized'] < df[mdl + '_VaR1']]

        plt.scatter(set1.index, set1, s=1, alpha=0.8, color=c[2], label="No break")
        plt.scatter(set2.index, set2, s=1, alpha=1, color=c[4], label="5% break")
        plt.scatter(set3.index, set3, s=1, alpha=1, color=c[3], label="1% break")

        c1 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.8, markersize=5, markerfacecolor=c[2])
        c2 = Line2D([0], [0], linestyle="none", marker="o", alpha=1, markersize=5, markerfacecolor=c[4])
        c3 = Line2D([0], [0], linestyle="none", marker="o", alpha=1, markersize=5, markerfacecolor=c[3])

        plt.hlines(0, datetime.datetime(1993, 1, 1), datetime.datetime(2014, 5, 1), alpha=0.6, lw=1)

        plt.xlim(datetime.datetime(1994, 12, 27), datetime.datetime(2013, 1, 1))
        plt.ylim(-0.11, 0)

        plt.legend((p1, p2, c1, c2, c3), ('VaR 5%', 'Var 1%', 'No break', '5% Break', '1% Break'), loc='lower left')
        ax = plt.gca()
        ax.set_yticklabels(['', '-10%', '-8%', '-6%', '-4%', '-2%', '0%'])
        if mdl == "nwrk":
            plt.title('Network model')
        else:
            plt.title('Benchmark model')
        plt.show()

        plt.savefig(mdl + '_VarLevels.pdf')
        plt.clf()


def graph_marginalDist(marginalReturn, names):
    mR = pd.DataFrame(marginalReturn)
    for x, n in enumerate(names):
        print mR.ix[:, x]

        seaborn.kdeplot(mR.ix[:, x], shade=True, color=c[0])
        plt.title(n)

        ax = plt.gca()
        ax.set_xticklabels(['', '-4%', '-2%', '0%', '+2%', '+4%'])
        ax.set_yticklabels([''])
        ax.set_ylim(0, 52.5)
        ax.set_xlim(0.945, 1.055)
        ax.legend_.remove()
        plt.savefig(n + "_marginal.pdf")
        plt.show()


def graph_pdf(dailyReturns):
    kde = sm.nonparametric.KDEUnivariate(dailyReturns)
    kde.fit()
    v1 = len(kde.cdf[kde.cdf<0.05])
    var = np.array(dailyReturns)[np.array(dailyReturns) < np.percentile(dailyReturns,5)]
    print var
    plt.vlines(kde.support[v1],0,100,colors=c[4],lw=1.5,linestyles='-', label='VaR 5%', zorder=5)
    plt.vlines(np.mean(var),0,100,colors=c[4],lw=1.5,linestyles=':', label='Expected shortfall 5%', zorder=5)

    plt.plot(kde.support[v1:], kde.density[v1:], color=c[1], alpha=1,zorder=2)
    plt.plot(kde.support[:v1], kde.density[:v1], color=c[3], alpha=1,zorder=2)
    plt.fill_between(kde.support[v1:], kde.density[v1:], edgecolor="None", color=c[1], alpha=0.55,zorder=3)
    plt.fill_between(kde.support[:v1], kde.density[:v1], edgecolor="None", color=c[3], alpha=0.55,zorder=3)
    ax = plt.gca()
    w=0.025
    ax.set_xlim(1-w-0.0005, 1+w+0.0005)
    ax.set_ylim(-3, 79)
    ax.set_xticklabels(['-3%', '-2%', '-1%', '0%', '+1%','+2%','+3%'])
    ax.set_yticklabels([''])
    plt.legend()

    #plt.show()
    plt.savefig('probdf.pdf')

    exit()

def BerkowitzGraph(data):
    print data.max()
    print data.min()
    bD = data[1].value_counts(sort=False)
    print bD



    seaborn.distplot(data,bins=100,kde=False,color=c[4],hist_kws={'edgecolor':'none'}, label="Percentile observations")

    plt.hlines(len(data)/100,-10,110,colors=c[1], linestyles='-', alpha=1, lw=1,label="Expected number of observations")
    plt.hlines(0,-10,110,colors='black', linestyles='-', alpha=0.5, lw=1)
    plt.ylim(-4,90)
    plt.xlim(1,100.9)
    plt.legend(loc="upper left")
    plt.title('Benchmark model')
    plt.savefig('berk_bench.pdf')
    plt.show()
    exit()


if __name__ == "__main__":
    # DecayBoot()
    # SOIovertime()
    BerkowitzGraph(pd.read_csv('berkowiz_bench2.csv',index_col=0,header=None,skiprows=1))
    #DecayBoot()