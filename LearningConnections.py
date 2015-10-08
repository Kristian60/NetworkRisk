import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
import matplotlib.pyplot as plt
from sklearn import covariance
import math
from scipy.optimize import minimize

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

def Optim(vals,df,market):

    lam = 0.1

    print len(vals)
    a = vals[:len(df.columns)]
    b = vals[len(df.columns):(len(df.columns))*2]
    w = vals[(len(df.columns))*2:].reshape((len(df.columns)),(len(df.columns)))

    total = 0
    for nr2,j in enumerate(df.columns):
        for nr1,i in enumerate(df.index):
            rhat = a[nr2] + b[nr2]*market[i] + sum([w[nr2,nr3]*(df.loc[i,j]-a[nr3]-b[nr3]*market[i]) for nr3 in range(len(df.columns)) if nr3 != nr2])
            total += math.exp(-(df.loc[i,j]-0.1)**2)*(df.loc[i,j]-rhat)**2 + lam*(a[nr2]**2+b[nr2]**2+np.linalg.det(w)**2)

    print total
    return total


def Learn():
    df = pd.read_csv('data.csv', sep=",")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna().ffill().set_index('Date')
    data = np.log(df).diff().dropna()[['Sydbank','DSV','Carlsberg B']]
    market = np.mean(data,axis = 1)
    x0 = np.zeros((len(data.columns))*2+(len(data.columns))**2)
    x0[len(data.columns):len(data.columns)*2] += 1
    print x0
    opt = minimize(Optim,x0,args=(data,market))

    a = opt.x[:len(data.columns)]
    b = opt.x[len(data.columns):(len(data.columns))*2]
    w = opt.x[(len(data.columns))*2:].reshape((len(data.columns)),(len(data.columns)))
    print "A"
    print a
    print
    print "B"
    print b
    print
    print "W"
    print w



    print data


if __name__ == "__main__":
    Learn()