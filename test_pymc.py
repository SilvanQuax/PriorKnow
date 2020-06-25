import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

# Initialize random number generator

# True parameter values
alpha, sigma = 1, 1
N = 100
# Size of dataset
n_input = 50
phi = np.linspace(-40.0, 40.0, n_input)
sig2N = 10
sig2P=25
muP=0

# Predictor variable

Sest= np.zeros(N)
Sreal = np.zeros(N)
Xall = np.zeros(N)

for ii in xrange(N):
    X1 = np.random.rand(1) * 10 - 5
    lamb = 3 * np.exp(- (X1 - phi) ** 2 / (2.0 * sig2N))
    R = np.random.poisson(lamb, size=(n_input)).astype('float32')

    basic_model = pm.Model()

    with basic_model:

        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=muP, sd=sig2P)

        # Expected value of outcome
        mu = 3*np.exp(-(alpha - phi) ** 2 / (2.0 * sig2N))

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Poisson('Y_obs', mu=mu, observed=R)

    map_estimate = pm.find_MAP(model=basic_model)
    a = np.ones(n_input) / sig2N
    e = phi / sig2N

    S = (np.dot(R, e) * sig2P + muP) / (sig2P * np.dot(R, a) + 1)
    Sest[ii] = map_estimate['alpha']
    Sreal[ii] = S
    Xall[ii] = X1
plt.figure();plt.plot(Sest);plt.plot(Sreal);plt.plot(Xall);plt.legend(['1','2','3'])
1

