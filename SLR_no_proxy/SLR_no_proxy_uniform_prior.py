import sys
import numpy as np
from math import exp, log
from scipy.stats import norm, halfcauchy, gamma, expon




def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

class SLR_BASIC:
    def __init__(self, y, x, act_params, iteration, burnin, thinning, add_chains=False):
        """
        Simple Linear Regression with measurement error
        implemented through gibbs sampling.

        Model
        y = beta0 + beta1 * x + epsilon
        where we observe
        w = x + v
        with hierarchical structures
        x ~ N(mu_x, sigma^2_x)
        v ~ N(0, sigma^2_v)
        epsilon ~ N(0, sigma^2_ep)

        Parameters
        ---------
        y: Response variable. n*1 dimension
        w: Observed covariate. n*1 dimension
        iteration: Number of iteration of MCMC, gibbs sampling
        burnin: Initial burnin periods of MCMC

        Returns
        -------
        """
        self.y = y
        self.n = len(y)
        one_vec = np.ones(self.n)
        self.X = np.c_[one_vec, x]
        self.k = self.X.shape[1]
        self.iteration = iteration
        self.burnin = burnin
        self.add_chains = add_chains
        self.thinning = thinning
        self.act_params = act_params

    def _init_prior(self):
        """
        Initialize priors for parameters
        beta0, beta1 ~ N(0, sigma^2_beta)
        mu_x ~ N(0, sigma^2_mu_x)
        sigma^2_x ~ IG(A_x, B_x)
        sigma^2_v ~ IG(A_v, B_v)
        sigma^2_ep ~ IG(A_ep, B_ep)
        Specify non-informative priors

        """

        print('======================Initialize MCMC======================')
        print(f'Total number of iterations: {self.iteration}')
        print(f'Burn-in periods: {self.burnin}')

        self.s2_beta = 10
        self.s2_mu_x = 10
        self.A_ep = self.B_ep = 0.1

        # params_name = ['beta0','beta1','mu_x', 's2_x', 's2_ep', 's2_v', 'x']
        #
        # self.params = {}
        # for name in params_name[:-1]:
        #     self.params[name] = np.zero(self.iteration)
        # self.params['x'] = np.zero((self.n, self.iteration))

        self.params = {}
        # initialize parameters
        self.params['beta0'] = [-2]
        self.params['beta1'] = [2]
        # self.params['s2_ep'] = [1]
        self.params['sigma_ep'] = [1]

    def _sampling_beta(self):
        sigma_ep = self.params['sigma_ep'][-1]
        s2_ep = sigma_ep**2
        s2_beta = self.s2_beta
        y = self.y
        X = self.X

        XTX = np.dot(X.T, X)

        # parameters of full conditionals
        temp = np.linalg.inv( XTX/s2_ep + np.identity(2)/s2_beta )
        mu = np.dot( temp ,
                     np.dot(X.T, y))/s2_ep
        cov = temp

        # mu = np.dot(np.linalg.inv(XTX) / s2_ep,  np.dot(X.T, y) / s2_beta ) / s2_ep
        # cov = np.linalg.inv(XTX) * s2_ep / 2

        # sampling from full conditionals
        beta0, beta1 = np.random.\
            multivariate_normal(mu, cov, 1).T

        # update posterior samples
        self.params['beta0'].append(beta0[0])
        self.params['beta1'].append(beta1[0])

    def _sampling_s2_ep(self):

        y = self.y
        X = self.X
        n = self.n

        beta0 = self.params['beta0'][-1]
        beta1 = self.params['beta1'][-1]
        beta = np.array([beta0, beta1])
        sigma1 = self.params['sigma_ep'][-1]

        def prop_dist(sigma):
            A=25
            termA = (1 + (sigma/A)**2)**(-1)
            termExp = -np.dot(np.dot(X, beta).reshape(self.n) - y ,
                              np.dot(X, beta).reshape(self.n) - y)/(2*sigma**2)
            return exp(termExp)*termA

        sigma_candidate = expon.rvs(scale=sigma1, size=1)[0]
        # numerator
        q_sigma1_given_sigma2 = expon.pdf(x=sigma1, scale=sigma_candidate)
        # denominator
        q_sigma2_given_sigma1 = expon.pdf(x=sigma_candidate, scale=sigma1)
        u = np.random.uniform(0,1,1)[0]

        acc_ratio =  prop_dist(sigma_candidate) * q_sigma1_given_sigma2 / \
                            (prop_dist(sigma1) * q_sigma2_given_sigma1)
        acc_prob = min([1, acc_ratio])

        if u < acc_prob:
            sigma2 = sigma_candidate
            self.params['sigma_ep'].append(sigma2)
        else:
            sigma2 = sigma1
            self.params['sigma_ep'].append(sigma2)



    def fit(self):

        self._init_prior()

        iteration = self.iteration

        for i in range(iteration):
            self.current_iteration = i
            self._sampling_beta()
            self._sampling_s2_ep()

            printProgress(i, iteration, 'Progress:', 'Complete', 1, 100)

    def mcmc_diagnose(self):

        return None