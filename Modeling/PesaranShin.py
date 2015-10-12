__author__ = 'Thomas'


import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
import matplotlib.pyplot as plt
from sklearn import covariance


pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)
def Main():
    H=10
    data = pd.read_csv("Data/Index_data.csv",sep=';')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    #data = data[['XLY','XLE','XLP']]
    model = sm.VAR(data)
    results = model.fit(maxlags=5,ic='aic')
    eps = results.resid
    SIGMA = np.cov(eps.T)
    ma_rep = results.ma_rep(maxn=H)
    nn = 0
    df = pd.DataFrame()
    print pd.DataFrame(SIGMA)
    for j in range(H):
        tt = SIGMA[nn,nn]**(-0.5)*(ma_rep[j,nn]).dot(SIGMA)
        print ma_rep[j,nn]
        for nr,i in enumerate(tt):
            df.loc[j,nr] = i
    for nr,j in enumerate(df.columns):
        plt.plot(df[j],label=data.columns[nr])
    plt.legend()
    plt.show()



if __name__ == "__main__":
    Main()