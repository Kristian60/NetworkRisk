__author__ = 'Thomas'

import pymc as pm
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def Main():
    data = np.array([np.random.normal(0,1) if np.random.uniform(0,1)<=0.1 else np.random.normal(10,0.5) for i in range(1000)])
    #plt.hist(data)
    #plt.show()
    p = pm.Uniform("p", 0, 1)
    assignment = pm.Categorical("assignment", [p, 1 - p], size=data.shape[0])
    taus = 1.0 / pm.Uniform("stds", 0, 100, size=2) ** 2
    centers = pm.Normal("centers", [8, 0], [2, 2], size=2)
    @pm.deterministic
    def center_i(assignment=assignment, centers=centers):
        return centers[assignment]

    @pm.deterministic
    def tau_i(assignment=assignment, taus=taus):
        return taus[assignment]

    # and to combine it with the observations:
    observations = pm.Normal("obs", center_i, tau_i, value=data, observed=True)

    # below we create a model class
    model = pm.Model([p, assignment, observations, taus, centers])
    mcmc = pm.MCMC(model)
    mcmc.sample(50000)
    center_trace = mcmc.trace("centers")[:]
    std_trace = mcmc.trace("stds")[:]
    p_trace = mcmc.trace("p")[:]
    pm.Matplot.plot(mcmc.trace("centers", 2), common_scale=False)
    plt.show()


    plt.hist(center_trace[:, 0], label="trace of center 0", color='blue')
    plt.hist(center_trace[:, 1], label="trace of center 1", color='red')
    plt.show()

    plt.hist(std_trace[:, 0], label="trace of std 0", color='blue')
    plt.hist(std_trace[:, 1], label="trace of std 1", color='red')
    plt.show()

    plt.plot(p_trace, label="trace of p", color='green')
    plt.show()

    plt.plot(center_trace[:, 0], label="trace of center 0", c='blue', lw=1)
    plt.plot(center_trace[:, 1], label="trace of center 1", c='red', lw=1)
    plt.title("Traces of unknown parameters")
    plt.show()



if __name__ == "__main__":
    Main()