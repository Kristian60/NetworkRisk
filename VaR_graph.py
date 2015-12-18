import numpy as np
import seaborn
import matplotlib.pyplot as plt
import scipy

c = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

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

scale=50

r = 0.1
d = -1.0

c1 = [0]*(4*scale)
c1.extend([1.1]*(16*scale))

plt.plot(np.linspace(0,0.2,20*scale),c1, lw=1, label='Portfolio value', color=c[1])
plt.fill_between(np.linspace(0,0.2,20*scale),c1,0, alpha=0.5, color=c[1])

plt.vlines(0.05, -1000, 1000, linestyles=':', lw=1)
plt.hlines(0,0,0.2,alpha=0.5)

plt.xlabel('Percentiles')
plt.ylabel('Portfolio value')
plt.legend()
plt.ylim([-0.1,1.5])
plt.xlim([-0.005,0.175])

plt.savefig('div_fig1.pdf')
plt.clf()


p0 = (0.96**2)
p2 = (0.04**2)
p1 = 1-p0-p2

c1 = [0]*(int(1*scale))
c1.extend([0.55]*int(7*scale))
c1.extend([1.1]*((20*scale)-len(c1)))


plt.plot(np.linspace(0,0.2,20*scale),c1, lw=1, label='Portfolio value', color=c[1])
plt.fill_between(np.linspace(0,0.2,20*scale),c1,0, alpha=0.5, color=c[1])

plt.vlines(0.05, -1000, 1000, linestyles=':', lw=1)
plt.hlines(0,0,0.2,alpha=0.5)
plt.legend()
plt.ylabel('Portfolio value')
plt.xlabel('Percentiles')
plt.ylim([-0.1,1.5])
plt.xlim([-0.005,0.175])
plt.savefig('div_fig2.pdf')
