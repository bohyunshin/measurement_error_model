import numpy as np
import pickle
from scipy.stats import expon, gamma, halfcauchy

def prop_dist(sigma):
    A = 25
    termA = (1 + (sigma / A) ** 2) ** (-1)
    return termA
initial_sigma = 2
mcmc_samples = []
mcmc_samples.append(initial_sigma)
iteration = 10000

for _ in range(iteration):
    sigma1 = mcmc_samples[-1]
    sigma_candidate = expon.rvs(scale=sigma1, size=1)[0]
    # numerator
    q_sigma1_given_sigma2 = expon.pdf(x=sigma1, scale=sigma_candidate)
    # denominator
    q_sigma2_given_sigma1 = expon.pdf(x=sigma_candidate, scale=sigma1)
    u = np.random.uniform(0, 1, 1)[0]

    acc_ratio = prop_dist(sigma_candidate) * q_sigma1_given_sigma2 / \
                (prop_dist(sigma1) * q_sigma2_given_sigma1)
    acc_prob = min([1, acc_ratio])

    if u < acc_prob:
        sigma2 = sigma_candidate
        mcmc_samples.append(sigma2)
    else:
        sigma2 = sigma1
        mcmc_samples.append(sigma2)

pickle.dump(mcmc_samples, open('../model/mcmc_cauchy.pkl','wb'))