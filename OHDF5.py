__author__ = 'Thomas'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import datetime
import os
import time
from scipy import stats

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

def GetFiles():
    '''

    Get All the files with relevant tickers

    :return:
    '''
    flist = sorted(os.listdir('Z:/TAQ/TAQHDF5/'))

    for ff in flist:
        if ff.replace('taq_','')[:4]>='2001' and ff.replace('taq_','')[:4]<'2014':
            print "Downloading..."
            t0=datetime.datetime.now()
            #ff = 'taq_20131231.h5'
            path = "Z:/TAQ/TAQHDF5/" + ff
            df = pd.read_hdf(path,'Trades')
            ind = pd.read_hdf(path,'TradeIndex')
            ind['end'] = np.cumsum(ind['count'])
            symlist = 'AAPL AXP BA CAT CSCO CVX DD DIS GE GS HD IBM INTC JNJ JOM KO MCD MMM MRK MSFT NKE PFE PG TRV UNH UTX V VZ WMT XOM'.split(' ')
            ind['ticker'] = [str(j).strip() for j in ind['ticker']]
            ind = ind[ind['ticker'].isin(symlist)].reset_index(drop=True)
            ran = np.array([range(start,end) for start,end in zip(ind['start'],ind['end'])])
            ran = [item for sublist in ran for item in sublist]
            df = df[df.index.isin(ran)]
            df['time'] = pd.to_datetime(df['utcsec'],unit='s')
            for i in ind.index:
                start = int(ind.loc[i,'start'])
                end = int(ind.loc[i,'end'])
                df.loc[start:end,'sym'] = ind.loc[i,'ticker']
            df.to_csv('data/taq/' + ff.replace('taq_','').replace('.h5','')+'.csv',columns=['time','price','sym'],index=False)
            print datetime.datetime.now()-t0
	
		


def SortFiles():
    '''
    Sort the content of the files and align structure of data frame
    :return:
    '''
    for ff in os.listdir('C:/Users/Thomas/Dropbox/UNI/Speciale/NetworkRisk/data/taqclean/'):
        #ff = 'data/taq/19930104.csv'
        print ff
        ddate = pd.to_datetime(ff.split('.')[0])
        df = pd.read_csv('C:/Users/Thomas/Dropbox/UNI/Speciale/NetworkRisk/data/taqclean/' + ff)
        temp = pd.DataFrame(index=[datetime.datetime(1970,1,1,9,30)+datetime.timedelta(0,0,0,0,j) for j in range(391)])
        temp.index.name = 'time'
        for j in np.unique(df['sym']):
            df['time'] = pd.to_datetime(df['time'])
            t = df[df['sym']==j].set_index('time').resample('Min',how='last')
            t.rename(columns={'price':t.loc[t.index[0],'sym']},inplace=True)
            t = t.drop('sym',1)
            temp = pd.merge(temp.reset_index(drop=False),t.reset_index(drop=False),'left',on='time').ffill().bfill().set_index('time')

        temp.index = [str(j).replace('1970-01-01',str(ddate).split(' ')[0]) for j in temp.index]
        temp.to_csv('data/taqagg/' + ff.split('/')[-1])

def AggFiles():
    '''

    Aggregate all the files into one single file

    :return:
    '''
    df = pd.DataFrame()
    for ff in os.listdir('C:/Users/Thomas/Dropbox/UNI/Speciale/NetworkRisk/data/taqagg/'):
        print ff
        temp = pd.read_csv('C:/Users/Thomas/Dropbox/UNI/Speciale/NetworkRisk/data/taqagg/' + ff)
        df = df.append(temp)
    df.rename(columns={'Unnamed: 0':'time'},inplace=True)
    df = df.set_index('time')
    df.to_csv('data/taq93-99.csv')

def BGallo():
    def BG_algo(nbrhd,d, y, obs):
        nbrhd.remove(obs)
        tmd_mean = stats.trim_mean(nbrhd, d)
        std = np.std(nbrhd)
        obs_dif = abs(obs-tmd_mean)
        acc = 3*std+y
        return obs_dif <= acc

    for ff in os.listdir('C:/Users/Thomas/Dropbox/UNI/Speciale/NetworkRisk/data/taq/'):
        if (ff) not in os.listdir('data/taqclean'):
            print ff, "CLEANING"
            t = pd.read_csv('C:/Users/Thomas/Dropbox/UNI/Speciale/NetworkRisk/data/taq/' + ff)
            t['time'] = pd.to_datetime(t['time'])
            k,d,y = 20,0.1,np.percentile(abs(t['price'].diff()).dropna(),95)
            fdf = pd.DataFrame()

            for j in np.unique(t['sym']):
                tdf = t[t['sym']==j].reset_index(drop=True)
                tdf = tdf.set_index('time').resample('Min',how='last').reset_index(drop=False).ffill().bfill()
                df = np.array(tdf['price'])
                remlist = []
                for n in range(len(df)):
                    if n <= k/2:
                        price = df[:k]
                    elif n >= (len(df)-(k/2)):
                        price = df[-k:]
                    else:
                        price = df[int(n-(k/2)):int(n+(k/2))]
                    if len(df)>1:
                        if BG_algo(list(price), d, y, df[n]) == False:
                            remlist.append(n)

                tdf = tdf[~tdf.index.isin(remlist)]
                tdf = tdf.ffill().bfill()
                fdf = fdf.append(tdf)

            fdf = fdf.set_index('time')
            fdf.to_csv('data/taqclean/'+ ff.split('/')[-1])

if __name__ == "__main__":
    BGallo()
    exit()

    #GetFiles()
    #BGallo()
    SortFiles()
    AggFiles()
