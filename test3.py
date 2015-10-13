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

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

utils = importr('utils')
pandas2ri.activate()
#utils.install_packages('Bessel')
#exit()



if __name__ == "__main__":
    t0 = datetime.datetime.now()
    data = pd.read_csv('data/minutedata2.csv',index_col=0)
    data.index = pd.to_datetime(data.index)
    data = np.log(data).diff().dropna().iloc[:,:]
    cols = data.columns
    data = pandas2ri.py2ri(data)
    rpy2.robjects.globalenv['data'] = data
    armaspec = (1,1,1,1,1,1)
    rscript = """
                suppressMessages(require(rugarch))
                suppressMessages(require(rmgarch))
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
    print b
    print datetime.datetime.now()-t0
    b.plot()
    plt.show()

