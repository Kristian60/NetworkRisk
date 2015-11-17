__author__ = 'Thomas'

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import pandas as pd
import seaborn

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)


def ICA_test():

    data = pd.read_csv('data/minutedata4.csv',index_col=0)
    data.index = pd.to_datetime(data.index)
    data = np.log(data).diff().dropna().iloc[:,:2]
    ind = np.random.normal(size=len(data))
    # Compute ICA
    ica = FastICA(n_components=len(data.columns))
    S_ = ica.fit_transform(data)  # Reconstruct signals
    print S_
    print np.dot(np.linalg.inv(ica.mixing_),data.T).T

    plt.plot(S_[:,0])
    #plt.plot(np.dot(np.linalg.inv(ica.mixing_),data.T).T[:,0])
    plt.plot(np.array(data)[:,0])
    plt.show()
    exit()



    plt.figure()

    models = [ind, S_, data]
    names = ['Observations (mixed signal)',
             'True Sources',
             'ICA recovered signals',
             'PCA recovered signals']
    colors = ['red', 'steelblue', 'orange']

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(3, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()

if __name__ == "__main__":
    ICA_test()