import pandas as pd
import numpy as np
import statsmodels.tsa.api as sm
import matplotlib.pyplot as plt
from sklearn import covariance
import seaborn as sns
import random
import scipy

pd.set_option('notebook_repr_html', True)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 3000)


def EstimateVAR(data, H):
    """

    :param data: A numpy array of log returns
    :param H: integer, size of step ahead forecast
    :return: a dataframe of connectivity or concentration parameters
    """

    model = sm.VAR(data)
    results = model.fit(maxlags=10, ic='aic')

    SIGMA = np.cov(results.resid.T)
    ma_rep = results.ma_rep(maxn=H)
    GVD = np.zeros_like(SIGMA)

    r, c = GVD.shape
    for i in range(r):
        for j in range(c):
            GVD[i, j] = 1 / np.sqrt(SIGMA[i, i]) * sum([ma_rep[h, i].dot(SIGMA[j]) ** 2 for h in range(H)]) / sum(
                [ma_rep[h, i, :].dot(SIGMA).dot(ma_rep[h, i, :]) for h in range(H)])
            # GVD[i,j] = SIGMAINV[i,i] * sum([ma_rep[h,i].dot(SIGMA[j])**2 for h in range(H)]) / sum([ma_rep[h,i,:].dot(SIGMA).dot(ma_rep[h,i,:]) for h in range(H)])
        GVD[i] /= GVD[i].sum()

    return pd.DataFrame(GVD), SIGMA, ma_rep, results.resid.T

 # test
#test22
# NU VIRKER DET
# NVD

def EstimateVAR_slow():
    df = pd.read_csv('C:/Users/thoru_000/Dropbox/Pers/PyCharmProjects/Speciale/data.csv', sep=";")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna().ffill().set_index('Date')
    data = np.log(df).diff().dropna()

    model = sm.VAR(data)
    results = model.fit(maxlags=5, ic='aic')

    coeff = results.coefs
    SIGMA = np.cov(results.resid.T)
    ma_rep = results.ma_rep(maxn=10)

    mse = results.mse(10)

    GVD = np.zeros_like(SIGMA)

    r, c = GVD.shape
    for i in range(r):
        for j in range(c):
            sel_j = np.zeros(r)
            sel_j[j] = 1
            sel_i = np.zeros(r)
            sel_i[i] = 1

            AuxSum = 0
            AuxSum_den = 0

            for h in range(10):
                AuxSum += (sel_i.T.dot(ma_rep[h]).dot(SIGMA).dot(sel_j)) ** 2
                AuxSum_den += (sel_i.T.dot(ma_rep[h]).dot(SIGMA).dot(ma_rep[h].T).dot(sel_i))

            GVD[i, j] = (AuxSum * (1 / SIGMA[i, i])) / AuxSum_den

        GVD[i] /= GVD[i].sum()

    pd.DataFrame(GVD).to_csv('GVD.csv', index=False, header=False)


def Bootstrap1p(sigma, iter):
    r = []
    b_r = []
    for i in range(iter):
        if i % (iter / 500.0) == 0:
            print i

        shock = np.array([random.choice(resid.T.values) for x in range(20)])
        p_r = [1] * 10
        for t, A in enumerate(marep[10::-1]):
            p_r *= shock[t, :].dot(marep[t]) + 1
            print p_r
            exit()

        r.append(sum([0.1 * a for a in p_r]))
        draw = random.choice(range(len(df) - 10))
        b_r.append(sum([0.1 * a for a in df.ix[draw, :] + 1]))

    dis = pd.DataFrame(np.array([r, b_r]).T, columns=["sim r", "br"])
    sns.distplot(dis['sim r'], label="sim", norm_hist=True)
    sns.distplot(dis['br'], label="Bootstrap", norm_hist=True)
    plt.legend()
    plt.show()


def BootstrapMult(resid, marep, iter):
    days = 10
    responseLength = 5

    a_col = np.zeros((iter, marep.shape[-1]))
    b_col = np.zeros((iter, marep.shape[-1]))

    for i in range(iter):
        if i % (iter / 500.0) == 0:
            print i

        shockMatrix = np.array([random.choice(resid.T.values) for x in range(responseLength * 2)])
        impulseResponseSystem = marep[::-1]

        returnPlaceholder = np.zeros((marep.shape[-1], marep.shape[-1]))

        for day in range(days):
            responsePlaceholder = np.zeros((impulseResponseSystem.shape[0], impulseResponseSystem.shape[1]))
            for responsePeriod in range(responseLength):

                shockVector = shockMatrix[responsePeriod + day, :]
                impulseResponseMatrix = impulseResponseSystem[responsePeriod]

                responsePlaceholder[responsePeriod] = shockVector.dot(impulseResponseMatrix)
            print pd.DataFrame(shockMatrix)
            print pd.DataFrame(responsePlaceholder)
            exit()

        a_col[i] = pd.DataFrame(returnPlaceholder + 1).product()

    exit()
    return a_col.flatten(), b_col.flatten()


if __name__ == "__main__":
    df = pd.read_csv('C:/Users/thoru_000/Dropbox/Speciale/Data/thesis-data.csv', sep=",", nrows=1000)
    df = df.set_index(pd.to_datetime(df['DATE'] + ' ' + df['TIME']))
    df = df.ix[:, 2:]
    df = np.log(df).diff().dropna()

    sns.distplot(df.values.flatten())
    plt.show()
    exit()
    con, sigma, marep, resid = EstimateVAR(df, 4)
    a, b = BootstrapMult(resid, marep, len(df))
    exit()
    # Bootstrap1p(sigma,100)
    df10 = pd.rolling_apply(df, 10, lambda x: np.prod(1 + x) - 1)
    df10 = df10.dropna().values.flatten()
    df10 = df10 - np.mean(df10) + 1
    print a.shape
    print b.shape
    sns.distplot(a, label='i.i.d.', norm_hist=True, bins=500)
    sns.distplot(b, label='Sequential', norm_hist=True, bins=500)
    sns.distplot(df10, label='Historical', norm_hist=True, bins=500)
    plt.legend()
    plt.show()
