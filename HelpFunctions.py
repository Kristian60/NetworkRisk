__author__ = 'Thomas'

import datetime
import pandas as pd
import numpy as np
import random
from scipy.stats import expon

def ExponBoot(data):
    '''

    :param data:
    :return: array with bootstrapped resids
    '''
    ### LAV 2 FUNKTIONER SÅ DET INDLEDENDE LAVES I EN FUNKTION UDEN FOR LOOPET MED ITERATIONER
    d1 = datetime.datetime(2013,03,1)
    shapeval = 10
    data = data[datetime.datetime(2013,3,1)-datetime.timedelta(50):'20130228']
    data['days_since'] = [(d1-j).days+1 for j in data.index]
    dsince = [1-expon.cdf(j,scale=shapeval) for j in list(np.unique(data['days_since']))]
    dsince /=np.sum(dsince)
    dsince = np.cumsum(sorted(dsince,reverse=False))
    dlist = sorted(np.unique(data['days_since']),reverse=True)
    klength = 390+15
    uninumbers = np.random.uniform(size=klength)
    a = [[j for j,i in zip(dlist,dsince) if uninumbers[k]<=i][0] for k in range(klength)]
    b = np.array([random.choice(np.array(data[data['days_since']==i].iloc[:,:-1])) for i in a])
    print pd.DataFrame(b)
    print data
    exit()
    return b


if __name__ == "__main__":
    data = pd.read_csv('data/minutedata4.csv',index_col=0)
    data.index = pd.to_datetime(data.index)
    data = np.log(data).diff().dropna()
    ExponBoot(data)
