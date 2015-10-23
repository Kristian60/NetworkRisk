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
    b = r(rscript)
    b = pd.DataFrame(b)
    b.plot()
    #plt.show()
    return b

def RCopula(data,sim):

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
    pd.DataFrame(b).iloc[:,1].plot()
    plt.show()
    exit()


    b = pd.DataFrame(b)
    return b

def RCopulaGarch(data,sim):
    data = pandas2ri.py2ri(data)
    rpy2.robjects.globalenv['data'] = data
    rpy2.robjects.globalenv['simulations'] = sim

    rscript = """
                suppressMessages(library(rugarch))
                suppressMessages(library(rmgarch))

                data <- matrix(rnorm(2200),200,11)
                nassets <- ncol(data)
                nperiods <- 390
                simulations <- 5000

                spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1),submodel = NULL,external.regressors = NULL, variance.targeting = FALSE),mean.model = list(armaOrder = c(1, 1), external.regressors = NULL,distribution.model = "norm", start.pars = list(), fixed.pars = list()))
                dccspec<-dccspec(uspec=multispec(replicate(ncol(data),spec)),dccOrder = c(1,1),distribution="mvnorm")
                mspec <-multispec(replicate(ncol(data),spec))
                cspec<-cgarchspec(mspec, VAR = FALSE, robust = FALSE, lag = 1, lag.max = NULL,lag.criterion = "AIC", external.regressors = NULL,robust.control = list(gamma = 0.25, delta = 0.01, nc = 10, ns = 500),dccOrder = c(1, 1), asymmetric = FALSE,distribution.model = list(copula = "mvt",method = "Kendall", time.varying = FALSE,transformation = "parametric"),start.pars = list(), fixed.pars = list())
                copgarch <- cgarchfit(cspec, data, spd.control = list(lower = 0.1, upper = 0.9, type = "pwm",kernel = "epanech"), fit.control = list(eval.se = TRUE, stationarity = TRUE,scale = FALSE), solver = "solnp", solver.control = list(), out.sample = 0,cluster = NULL, fit = NULL, VAR.fit = NULL, realizedVol = NULL)
                simfit <- cgarchsim(copgarch, n.sim = nperiods, n.start = 0, m.sim = simulations,startMethod = "sample", presigma = NULL, preresiduals = NULL,prereturns = NULL, preR = NULL, preQ = NULL, preZ = NULL, rseed = NULL,mexsimdata = NULL, vexsimdata = NULL, cluster = NULL, only.density = FALSE,prerealized = NULL)
                simdata <- fitted(simfit)

                t <- array(rep(nperiods*nassets*simulations),c(nperiods,nassets,simulations))
                for (i in 1:simulations) {
                  t[,,i] <- fitted(simfit,i)
                }
                t

                 """

    print rscript
    b = r(rscript)
    return b



if __name__ == "__main__":
    t0 = datetime.datetime.now()
    data = pd.read_csv('data/minutedata2.csv',index_col=0)
    data.index = pd.to_datetime(data.index)
    #data = np.log(data).diff().dropna().iloc[:int(6.5*60*5),:]
    data = np.log(data).diff().dropna().iloc[:10000,:]
    sim = RCopulaGarch(data,1000)
    print sim
    print datetime.datetime.now()-t0


