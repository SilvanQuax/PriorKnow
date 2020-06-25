import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

lower, upper = -20,20
mu, sigma = 0., np.sqrt(16.)
X = stats.truncnorm((lower-mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N=stats.norm(loc=mu, scale=sigma)
fig, ax = plt.subplots(2, sharex=True)
ax[0].hist(X.rvs(10000), bins=20, normed=True)
ax[1].hist(N.rvs(10000), bins = 20, normed=True)
plt.xlim([-30,30])
plt.show()

n_input=50
n_samples=1000
# g = (np.linspace(0.5, 4, 6)).tolist()
g =(np.linspace(0.2,1.6,6)).tolist()
t_samples = n_samples * len(g)

phi = np.linspace(-40.0, 40.0, n_input)

TG = stats.truncnorm((lower - mu) / sigma, (upper - mu) /sigma, loc=mu, scale=sigma)
X = TG.rvs([t_samples, 1])

X = np.tile(X, (1, n_input))
G = np.tile(g, (1, n_samples)).T
lamb = G * np.exp(- (X - np.tile(phi, (t_samples, 1))) ** 2 / (2.0 * 10))

R = np.random.poisson(lamb, size=(t_samples, n_input)).astype('float32')
plt.figure()
plt.hist(np.sum(R,1), bins=20, normed=True)
plt.show()
1

plt.figure(1)
X=np.linspace(-20,20,1000)
sig2N = 36
plt.plot(X,np.exp(-X**2/(2*sig2N)))
plt.show()