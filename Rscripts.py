__author__ = 'Thomas'

import pandas as pd
import numpy as np
import rpy2.robjects
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import matplotlib.pyplot as plt
import seaborn
import datetime
import scipy.stats



pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

pandas2ri.activate()

install = 0

if install == 1:
    utils = importr('utils')
    utils.install_packages('copula',dependencies=True)
    r('install.packages("C:/Users/Thomas/Downloads/gsl_1.9-10.tar.gz", repos = NULL, type = "source")')
    #utils.install_packages("gsl")
    exit()

def RDCC(data):
    cols = data.columns
    data = pandas2ri.py2ri(data)
    rpy2.robjects.globalenv['data'] = data
    armaspec = (1,1,1,1,1,1)

    rscript = """
                suppressMessages(library(rugarch))
                suppressMessages(library(rmgarch))
                ###data <- matrix(rnorm(2200),200,11)
                spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(%s, %s),submodel = NULL,
                                                         external.regressors = NULL, variance.targeting = FALSE),
                                   mean.model = list(armaOrder = c(%s, %s), external.regressors = NULL,
                                                     distribution.model = "norm", start.pars = list(), fixed.pars = list()))
                dccspec<-dccspec(uspec=multispec(replicate(11,spec)),dccOrder = c(%s,%s),distribution="mvnorm")
                dccgarch<-dccfit(dccspec,data = data)
                dccsimdata<-dccsim(dccgarch,n.sim=1000)
                dccgarch
                fitted(dccsimdata)
                 """ %(armaspec[0],armaspec[1],armaspec[2],armaspec[3],armaspec[4],armaspec[5])

    print rscript
    test = r(rscript)
    b = pd.DataFrame(test,columns=cols)
def RCopula(data,N):

    cols = data.columns
    data2 = pandas2ri.py2ri(data)
    data = np.array(data)
    rpy2.robjects.globalenv['data'] = data2
    rpy2.robjects.globalenv['N'] = N


    rscript = """ suppressMessages(library(copula))

                    nAssets <- ncol(data)
                    u <- pobs(data,N)
                    clayton.cop <- claytonCopula(2,dim=nAssets)
                    a <- fitCopula(clayton.cop,u,method="mpl")
                    y <- (rCopula(copula=claytonCopula(a@estimate,nAssets),n=N))
                    y"""

    print rscript


    b = r(rscript)

    ###########????????????
    for j in range(b.shape[1]):
        mean = np.mean(data[:,j])
        std = np.std(data[:,j])
        for i in range(b.shape[0]):
            b[i,j] = scipy.stats.norm.ppf(b[i,j],loc=mean,scale=std)

    seaborn.distplot(b.ravel(),norm_hist=True)
    print scipy.stats.kurtosis(b.ravel())
    print scipy.stats.skew(b.ravel())
    print np.mean(b.ravel())
    print np.std(b.ravel())
    plt.show()
    exit()


    b = pd.DataFrame(b)
    return b

if __name__ == "__main__":
    t0 = datetime.datetime.now()
    data = pd.read_csv('data/minutedata2.csv',index_col=0)
    data.index = pd.to_datetime(data.index)
    data = np.log(data).diff().dropna().iloc[:int(6.5*60*5),:]
    sim = RCopula(data,10000)
    print sim
    print datetime.datetime.now()-t0


