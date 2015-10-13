from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from Connectedness import EstimateVAR
import pandas as pd
import random
import seaborn
import datetime
import statsmodels.tsa.api as sm
from statsmodels.tsa.vector_ar.var_model import ma_rep

def Main():
    pass


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


def VarSimul(data,H):
    model = sm.VAR(data)
    results = model.fit(H)
    VARcoeff = results.params[1:]
    VARcoeff = np.array(VARcoeff).reshape(len(VARcoeff)/len(data.columns),len(data.columns),len(data.columns))
    VARstd = results.stderr[1:]
    VARstd = np.array(VARstd).reshape(len(VARstd)/len(data.columns),len(data.columns),len(data.columns))
    test = []
    print VARcoeff[3][0,6]
    for i in range(1000):
        if i%1000==0:
            print i
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
        print ddate
        td.index = pd.to_datetime(td.index)
        td = td[(td.index.month==ddate.month) & (td.index.day==ddate.day) & (td.index.year==ddate.year)]
        if len(td)>0:
            gvd,sigma,marep, resid = EstimateVAR(td,15)
            soi = 0
            for i in gvd.index:
                for j in gvd.columns:
                    if i!=j:
                        soi += gvd.loc[i,j]
            soi /= len(gvd)
            months.append(td.index[-1])
            sois.append(soi)
            #print soi
    plt.plot_date(months,sois,fmt='-')
    plt.title('Total Spillover Index')
    plt.xlabel('Date')
    plt.ylabel('SOI')
    plt.show()



if __name__ == "__main__":
    data = pd.read_csv('data/minutedata2.csv',index_col=0)
    data.index = pd.to_datetime(data.index)
    data = np.log(data).diff().dropna()
    SOI(data,15)




