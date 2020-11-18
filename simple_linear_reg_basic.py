import sys
import numpy as np
from scipy.stats import invgamma




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

        self.s2_beta = 10**8
        self.s2_mu_x = 10**8
        self.A_ep = self.B_ep = 0.001

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
        self.params['s2_ep'] = [1]

    def _sampling_beta(self):
        s2_ep = self.params['s2_ep'][-1]
        s2_beta = self.s2_beta
        y = self.y
        X = self.X

        XTX = np.dot(X.T, X)

        # parameters of full conditionals
        temp = np.linalg.inv( XTX/s2_ep + np.identity(2)/s2_beta )
        mu = np.dot( temp ,
                     np.dot(X.T, y))/s2_ep
        cov = temp
        # sampling from full conditionals
        beta0, beta1 = np.random.\
            multivariate_normal(mu, cov, 1).T

        # update posterior samples
        self.params['beta0'].append(beta0[0])
        self.params['beta1'].append(beta1[0])

    def _sampling_s2_ep(self):

        A_ep = self.A_ep
        B_ep = self.B_ep

        y = self.y
        X = self.X

        beta0 = self.params['beta0'][-1]
        beta1 = self.params['beta1'][-1]
        beta = np.array([beta0, beta1])

        # parameters of full conditionals
        shape = A_ep + self.n/2
        scale = B_ep + np.dot(np.dot(X, beta).reshape(self.n) - y ,
                              np.dot(X, beta).reshape(self.n) - y)/2

        # sampling from full conditionals
        s2_ep = invgamma.rvs(a=shape, scale=scale, size=1)[0]

        # update posterior samples
        self.params['s2_ep'].append(s2_ep)


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