__author__ = 'Thomas'

import numpy as np
from sklearn import covariance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from graphical_lasso import graphical_lasso

def Main():
    cov = np.array([[0.5,0.2,0.1],[0.2,0.6,0.09],[0.1,0.09,0.7]])
    data = np.random.multivariate_normal([0,0,0],cov,5000)
    rows = 5
    cls = 5
    fig1,axs = plt.subplots(rows,cls,figsize=(20,16),facecolor='grey',edgecolor = 'black')
    fig1.subplots_adjust(hspace = .5, wspace=.5)
    axs = axs.ravel()

    for cc in range(rows*cls):
        cov = graphical_lasso(data,alpha=0.01*cc)
        print cov[0]
        seaborn.heatmap(pd.DataFrame(np.array(cov[0])),ax=axs[cc])
        axs[cc].set_title(cc)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Main()