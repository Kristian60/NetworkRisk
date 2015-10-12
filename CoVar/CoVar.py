__author__ = 'Thomas'

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn


pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

def CoVar():
    df = pd.read_csv("Data/Index_data.csv",sep=';')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna().ffill().set_index('Date')
    #data = np.log(df).diff().dropna()[['Nordea Bank','Sydbank','Danske Bank','Jyske Bank','Novo Nordisk B']]
    data = np.log(df).diff().dropna()
    data = data.rename(columns={'S&P':'SP'})
    data = data.replace(np.inf,0).replace(-np.inf,0)
    mCovar = np.zeros((len(data.columns),len(data.columns)))


    q = 0.05
    for nr1,j in enumerate(data.columns):
        for nr2,k in enumerate(data.columns):
            if nr1 != nr2:
                mod = smf.quantreg(str(j) + '~' + str(k),data)
                res = mod.fit(q=q)
                var5 = mod.fit(q=q).params[0]
                var5 = np.percentile(data[k],q)
                var50 = mod.fit(q=0.5).params[0]
                covar = res.params[0] + res.params[1]*var5
                print covar
                print res.summary()
                dcovar = mod.fit(q=q).params[1]*(var5-var50)
                #res = mod.fit(q=0.5)
                mCovar[nr1,nr2] = round(dcovar,3)
                icepts = []
                for i in np.arange(0.01,1,0.01):
                    res = mod.fit(q=i)
                    icepts.append(res.params[1])
                plt.plot(np.arange(0.01,1,0.01),icepts)
                #plt.ylim(0,1)
                print j,k
                plt.show()
            else:
                mCovar[nr1,nr2] = np.nan



            #mCovar[nr1,nr2]
    print pd.DataFrame(mCovar,columns=data.columns,index=data.columns)


if __name__ == "__main__":
    CoVar()