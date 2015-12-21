
# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause

import datetime
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_historical_yahoo
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold
import pandas as pd
import pandas.io.data as web
from graphical_lasso import graphical_lasso
import Quandl
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, WeekdayLocator


pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)

def StockMarketOLD():
    ###############################################################################
    # Retrieve the data from Internet

    # Choose a time period reasonnably calm (not too long ago so that we get
    # high-tech firms, and before the 2008 crash)
    d1 = datetime.datetime(2005, 1, 1)
    d2 = datetime.datetime(2009, 12, 31)

    # kraft symbol has now changed from KFT to MDLZ in yahoo
    symbol_dict = {
        'TOT': 'Total',
        'XOM': 'Exxon',
        'CVX': 'Chevron',
        'COP': 'ConocoPhillips',
        'VLO': 'Valero Energy',
        'MSFT': 'Microsoft',
        'IBM': 'IBM',
        'TWX': 'Time Warner',
        'CMCSA': 'Comcast',
        #'CVC': 'Cablevision',
        #'YHOO': 'Yahoo',
        #'DELL': 'Dell',
        'HPQ': 'HP',
        'AMZN': 'Amazon',
        'TM': 'Toyota',
        'CAJ': 'Canon',
        'MTU': 'Mitsubishi',
        'SNE': 'Sony',
        #'F': 'Ford',
        'HMC': 'Honda',
        #'NAV': 'Navistar',
        'NOC': 'Northrop Grumman',
        'BA': 'Boeing',
        'KO': 'Coca Cola',
        'MMM': '3M',
        'MCD': 'Mc Donalds',
        #'PEP': 'Pepsi',
        'MDLZ': 'Kraft Foods',
        'K': 'Kellogg',
        'UN': 'Unilever',
        'MAR': 'Marriott',
        'PG': 'Procter Gamble',
        'CL': 'Colgate-Palmolive',
        'GE': 'General Electrics',
        'WFC': 'Wells Fargo',
        'JPM': 'JPMorgan Chase',
        #'AIG': 'AIG',
        'AXP': 'American Express',
        'BAC': 'Bank of America',
        'GS': 'Goldman Sachs',
        'AAPL': 'Apple',
        'SAP': 'SAP',
        'CSCO': 'Cisco',
        'TXN': 'Texas Instruments',
        'XRX': 'Xerox',
        #'LMT': 'Lookheed Martin',
        'WMT': 'Wal-Mart',
        'WBA': 'Walgreen',
        'HD': 'Home Depot',
        'GSK': 'GlaxoSmithKline',
        'PFE': 'Pfizer',
        'SNY': 'Sanofi-Aventis',
        'NVS': 'Novartis',
        'KMB': 'Kimberly-Clark',
        'R': 'Ryder',
        'GD': 'General Dynamics',
        'RTN': 'Raytheon',
        'CVS': 'CVS',
        'CAT': 'Caterpillar',
        'DD': 'DuPont de Nemours',

        #'GM': 'General Motors',
        #'GOOG' : 'Google',
        'ORCL' : 'Oracle',
        'NVO':'Novo Nordisk',
        'LLY':'Eli Lilly and Company',
        #'FB':'Facebook',
        'MRK':'Merck Co',
        }
    '''
    symbol_dict = {'Danske.CO':'Danske Bank',
                   'Maersk-B.CO':'Maersk',
                   'DSV.CO':'DSV',
                   'FLS.CO':'FLS',
                   'Gen.CO':'Genmab',
                   'TDC.CO':'TDC',
                   'CARL-B.CO':'Carlsberg',
                   'CHR.CO':'Chr Hansen',
                   'COLO-B.CO':'Coloplast',
                   'GN.CO':'GN Store Nord',
                   'NDA-DKK.co':'Nordea',
                   'Novo-B.co':'Novo Nordisk',
                   'NZYM-B.CO':'Novozymes',
                   'PNDORA.CO':'Pandora',
                   'Tryg.co':'Tryg',
                   'VWS.CO':'Vestas',
                   'WDH.CO':'William Demant',
                   'G4s.co':'G4S',
                   'JYSK.CO':'Jyske Bank',
                   'KBHL.CO':'Kobenhavns Lufthavne',
                   'RBREW.CO':'Royal Unibrew',
                   'ROCK-B.CO':'Rockwool',
                   'SYDB.CO':'Sydbank',
                   'TOP.CO':'Topdanmark',
                   #'ALMB.CO':'Alm Brand',
                   'AURI-B.CO':'Auriga',
                   'Bava.CO':'Bavarian Nordic',
                   'BO.CO':'Bang Olufsen',
                   'DFDS.CO':'DFDS',
                   'DNORD.CO':'DS Norden',
                   'GES.CO':'Greentech',
                   'IC.CO':'IC Group',
                   'JDAN.CO':'Jeudan',
                   #'JUTBK.CO':'Jutlander Bank',
                   #'MATAS.CO':'Matas',
                   'NKT.CO':'NKT',
                   #'NNIT.CO':'NNIT',
                   'NORDJB.CO':'Nordjyske Bank',
                   #'ONXEO.CO':'Onxeo',
                   #'OSSR.CO':'Ossur',
                   'PAAL-B.CO':'Per Aarslef',
                   'RILBA.CO':'Ringkobing Landbobank',
                   'SAS-DKK.CO':'SAS',
                   'SCHO.CO':'Schouw Co.',
                   'SIM.CO':'SimCorp',
                   'Solar-B.co':'Solar B',
                   'SPNO.CO':'Spar Nord',
                   'TIV.CO':'Tivoli',
                   'UIE.CO':'UIE',
                   'VELO.CO':'Veloxis',
                   'ZEAL.CO':'Zealand Pharma'
                   }
    '''
    symbols, names = np.array(list(symbol_dict.items())).T

    for symbol in symbols:
        print symbol
        if len(pd.DataFrame(np.array([[q[5] for q in quotes_historical_yahoo(symbol,d1,d2,True,False)]]).T)) != 1259:
            print symbol, len(pd.DataFrame(np.array([[q[5] for q in quotes_historical_yahoo(symbol,d1,d2,True,False)]]).T))


    open = pd.DataFrame(np.array([[q[5] for q in quotes_historical_yahoo(symbol,d1,d2,True,False)] for symbol in symbols]).T)
    close = pd.DataFrame(np.array([[q[6] for q in quotes_historical_yahoo(symbol,d1,d2,True,False)] for symbol in symbols]).T)

    # The daily variations of the quotes are what carry most information
    variation = np.array(close - open)

    ###############################################################################
    # Learn a graphical structure from the correlations
    #edge_model = covariance.GraphLassoCV()


    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery


    df = pd.read_csv('data/TData9313_final5.csv',index_col=0)
    X = variation.copy()

    pd.DataFrame(np.round(np.cov(X.T),3),columns=symbols,index=symbols).to_latex('covariancetable.tex')

    print np.max(np.round(np.cov(X.T),3))

    X /= X.std(axis=0)

    covariance_,precision_ = graphical_lasso(X,0.3)

    print pd.DataFrame(precision_)

    #edge_model.fit(X)

    ###############################################################################
    # Cluster using affinity propagation

    _, labels = cluster.affinity_propagation(covariance_)

    n_labels = labels.max()

    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(symbols[labels == i])))

    ###############################################################################
    # Find a low-dimension embedding for visualization: find the best position of
    # the nodes (the stocks) on a 2D plane

    # We use a dense eigen_solver to achieve reproducibility (arpack is
    # initiated with random vectors that we don't control). In addition, we
    # use a large number of neighbors to capture the large-scale structure.
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver='dense', n_neighbors=6)

    embedding = node_position_model.fit_transform(X.T).T

    ###############################################################################
    # Visualization
    plt.figure(1, facecolor='w', figsize=(20, 16))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    plt.annotate('From %s to %s' % (d1.strftime('%Y-%m-%d'),d2.strftime('%Y-%m-%d')),xy=(0.11,-0.37),size=25)

    print X.shape

    for i in range(n_labels + 1):
        plt.annotate('Cluster %i: %s' % ((i + 1), ', '.join(symbols[labels == i])),xy=(-0.43,0.02-i*0.02),size=18)
        pass



    # Display a graph of the partial correlations
    #partial_correlations = edge_model.precision_.copy()
    partial_correlations = precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=200 * d ** 2, c=labels,
                cmap=plt.cm.spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    #a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.get_cmap('Greys'),
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=22,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.spectral(label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .25 * embedding[0].ptp(),
             embedding[0].max() + .20 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .20 * embedding[1].ptp(),
             embedding[1].max() + .20 * embedding[1].ptp())

    plt.savefig('Graphs/StockCluster.pdf',bbox_inches='tight')
    plt.savefig('Graphs/StockCluster.svg',bbox_inches='tight')
    plt.show()


def StockMarket():

    X = pd.read_csv('data/TData9313_final5.csv',index_col=0)
    X.index = pd.to_datetime(X.index)
    X = X['20080101':'20091231']
    X = X.resample('B',how='last').ffill().dropna(axis=0,how='all').dropna(axis=1)
    #X = X.dropna(axis=1)
    X = np.log(X).diff()[1:]
    pd.DataFrame(np.round(np.cov(X.T),5),columns=X.columns,index=X.columns).to_latex('covariancetable.tex')

    print pd.DataFrame(np.round(np.cov(X.T),5),columns=X.columns,index=X.columns).max()
    print pd.DataFrame(np.round(np.cov(X.T),5),columns=X.columns,index=X.columns)
    print pd.DataFrame(np.round(np.cov(X.T),5),columns=X.columns,index=X.columns).idxmax()


    X /= X.std(axis=0)

    covariance_,precision_ = graphical_lasso(X,0.6)
    print pd.DataFrame(precision_)

    gephi = pd.DataFrame()
    nodes = pd.DataFrame()
    nr = 0
    artdf = pd.DataFrame(precision_,columns=X.columns,index=X.columns)
    artdf2 = pd.DataFrame(covariance_,columns=X.columns,index=X.columns)
    print "Create Gephi Files"
    for nr1,i in enumerate(artdf.index):
        for nr2,j in enumerate(artdf.columns):
            if nr1 > nr2:
                if abs(artdf.loc[i,j]) > 0:
                    gephi.loc[nr,'Source'] = nr1+1
                    gephi.loc[nr,'Target'] = nr2+1
                    gephi.loc[nr,'Type'] = 'Undirected'
                    gephi.loc[nr,'Id'] = nr+1
                    gephi.loc[nr,'Label'] = nr1+1
                    gephi.loc[nr,'Weight'] = abs(artdf.loc[i,j])
                    gephi.loc[nr,'Weight'] = abs(artdf2.loc[i,j])
                    gephi.loc[nr,'name'] = str(i)
                    gephi.loc[nr,'toname'] = str(j)
                    nr += 1
        nodes.loc[nr1,'Nodes'] = nr1+1
        nodes.loc[nr1,'Id'] = nr1+1
        nodes.loc[nr1,'Label'] = str(i).split('feat')[0]



    gephi.to_csv('gephi.csv')
    nodes.to_csv('nodes.csv')


def OilPrice():
    dfomat = DateFormatter('%Y')
    years = YearLocator()   # every year
    df = Quandl.get("ODA/POILBRE_USD")['20080101':'20091231']
    fig,ax = plt.subplots()
    ax.plot_date(df.index,df['Value'],fmt='-',color=seaborn.xkcd_rgb['black'])
    #ax.xaxis.set_major_locator(years)
    #ax.xaxis.set_major_formatter(dformat)
    plt.ylim(0,150)
    plt.ylabel('US Dollar per Barrel of Crude Oil')
    plt.savefig('Graphs/CrudeOil.pdf',bbox_inches='tight')
    plt.show()
    print df


if __name__ == "__main__":

    seaborn.set(context='paper', rc={
        'axes.facecolor': '#F0F0F0',
        'figure.facecolor': '#F0F0F0',
        'savefig.facecolor': '#F0F0F0',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'grid.color': '#DADADA',
        'ytick.color': '#66666A',
        'xtick.color': '#66666A'
    })


    OilPrice()