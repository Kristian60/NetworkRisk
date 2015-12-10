import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

scale=50
r = 0.1
d = -1.0

c1 = [0]*(4*scale)
c1.extend([1.1]*(16*scale))

plt.plot(np.linspace(0,0.2,20*scale),c1, lw=2, label='Portfolio value')
plt.fill_between(np.linspace(0,0.2,20*scale),c1,0, alpha=0.5)

plt.vlines(0.05, -1000, 1000, linestyles='--')
plt.hlines(0,0,0.2,alpha=0.5)

plt.xlabel('Percentiles')
plt.ylabel('Portfolio value')
plt.legend()
plt.ylim([0,1.5])

plt.savefig('div_fig1.pdf')
plt.clf()


p0 = (0.96**2)
p2 = (0.04**2)
p1 = 1-p0-p2

c1 = [0]*(int(1*scale))
c1.extend([0.55]*int(7*scale))
c1.extend([1.1]*((20*scale)-len(c1)))


plt.plot(np.linspace(0,0.2,20*scale),c1, lw=2, label='Portfolio value')
plt.fill_between(np.linspace(0,0.2,20*scale),c1,0, alpha=0.5)

plt.vlines(0.05, -1000, 1000, linestyles='--')
plt.hlines(0,0,0.2,alpha=0.5)
plt.legend()
plt.ylabel('Portfolio value')
plt.xlabel('Percentiles')
plt.ylim([0,1.5])
plt.savefig('div_fig2.pdf')
