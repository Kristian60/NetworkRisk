__author__ = 'Thomas'

import pandas as pd
import numpy as np
import pandas.io.data as web
import datetime
import seaborn
import matplotlib.pyplot as plt
import statsmodels.tsa.api as sm

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)


def GetData():
    start = datetime.datetime(2000,1,1)
    end = datetime.datetime(2014,12,31)
    df = pd.DataFrame()
    companies ={'Carlsberg A':'CARL-A.CO',
                'Carlsberg B':'CARL-B.CO',
                'Chr Hansen':'CHR.CO',
                'Coloplast':'COLO-B.CO',
                'Danske Bank':'DANSKE.CO',
                'DSV':'DSV.CO',
                'FLSmidth':'FLS.CO',
                'G4S':'G4S.CO',
                'Genmab':'GEN.CO',
                'GN Store Nord':'GN.CO',
                'ISS':'ISS.CO',
                'Jyske Bank':'JYSK.CO',
                'Koebenhavns Lufthavne':'KBHL.CO',
                'Lundbeck':'LUN.CO',
                'A.P. Moeller Maersk A':'MAERSK-A.CO',
                'A.P. Moeller Maersk B':'MAERSK-B.CO',
                'Nordea Bank':'NDA-DKK.CO',
                'Novo Nordisk B':'NOVO-B.CO',
                'Novozymes B':'NZYM-B.CO',
                'Pandora':'PNDORA.CO',
                'Royal UNIBREW':'RBREW.CO',
                'Rockwool Int. A':'ROCK-A.CO',
                'Rockwool Int. B':'ROCK-B.CO',
                'Sydbank':'SYDB.CO',
                'TDC':'TDC.CO',
                'Topdanmark':'TOP',
                'Tryg':'TRYG.CO',
                'Vestas Wind Systems':'VWS',
                'William Demant Holding':'WDH'}
    for comp in companies:
        print comp
        f = web.DataReader(companies[comp],'yahoo',start,end)[['Adj Close']].reset_index() ### Used since it adjusts for dividend and splits
        f.columns = ['Date',comp]
        print f
        exit()
        if len(df)==0:
            df = f
        else:
            df = pd.merge(df,f,on='Date',how='outer')
    for j in df.columns:
        #plt.plot_date(df.index,df[j],fmt='-')
        pass
    df = df.sort('Date')
    df = df.set_index('Date')
    #plt.show()
    print df
    df.to_csv('data.csv')
    return df
def EstimateVAR(df):
    df = pd.read_csv('data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna().ffill().set_index('Date')
    #df = df.drop('William Demant Holding',1)

    data = np.log(df).diff().dropna()
    df.to_csv('data.csv')
    model = sm.VAR(data)
    results = model.fit(maxlags=5,ic='aic')
    fevd = results.fevd(10)
    print fevd.summary()
    allcomp = fevd.decomp[:,-1]
    for i,name in zip(fevd.decomp,fevd.names):
        print name
        tempdf = pd.DataFrame(i,columns=fevd.names)
        tempdf.to_csv('test/' + name + '.csv')
    exit()
def FormatData():
    df = pd.read_csv('data/minutedata3.csv',index_col = 0)
    df.index = pd.to_datetime(df.index,format="%d/%m/%Y %H:%M")
    df.to_csv('data/minutedata4.csv')
    print df

if __name__ == "__main__":
    FormatData()