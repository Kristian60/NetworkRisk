import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import datetime
import Connectedness

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

f = pd.read_csv('data/taq93.csv', index_col=0)

if __name__ == "__main__":
    c = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    seaborn.set(context='paper', font='Segoe UI', font_scale=0.3, rc={
        'axes.facecolor': '#F0F0F0',
        'figure.facecolor': '#F0F0F0',
        'savefig.facecolor': '#F0F0F0',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'grid.color': '#DADADA',
        'ytick.color': '#66666A',
        'xtick.color': '#66666A',
        'grid.linewidth': 0.2,
    })

    #relevant = 'AAPL AXP BA CSCO CVX GS IBM MSFT PG V XOM'.split(' ')

    ndf = pd.read_csv('data/TData9313_final5.csv', sep=",", index_col=0, nrows=1)
    df = pd.read_csv('data/TData9313_final5.csv', sep=",", index_col=0, skiprows=2000000, nrows=30000,
                     names=ndf.columns)
    #df = df[relevant]

    df.index = pd.to_datetime(df.index)
    daily = df.resample('d', how='last').dropna(how='all')

    df = np.log(df).diff().dropna(axis=1, how='all').dropna()


    _, _SIGMA, MA, _results = Connectedness.EstimateVAR(df, 15)



    f, ax = plt.subplots(df.shape[1], df.shape[1], figsize=(11,7.5))

    for n, x in enumerate(ax):
        for q, z in enumerate(x):
            if not q == n:
                z.axhline(0, 0, 10, linewidth=0.1, color='black')
                z.axvline(0, 0, 10, linewidth=0.1, color='black')
                z.plot(MA[0:, q, n], linewidth=0.3, color='black')
                z.set_xlim(-1, 15)
                z.set_ylim(-0.15, 0.15)

            else:
                f.delaxes(z)

            if q == 0:
                z.set_ylabel(df.columns[n], rotation=90)

            if n == df.shape[1] - 1:
                z.set_xlabel(df.columns[q])

            if q == 0 and n == df.shape[1] - 1:
                print "yay"
                z.set_yticklabels(['0.15', '', '0.5', '', '-0.5', '', '-0.15'])
                pass
            else:
                z.set_yticklabels([])
                z.set_xticklabels([])
    plt.tight_layout(pad=2, h_pad=0.3, w_pad=0.3)
    #plt.show()
    plt.savefig('ImpulseResponse_all.pdf',bbox_inches='tight')
