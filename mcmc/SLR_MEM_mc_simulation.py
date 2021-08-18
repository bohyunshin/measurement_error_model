import numpy as np
from simple_linear_reg import SLR_ME
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# generate toy data
beta0 = -1
beta1 = 1
mu_x = 1/2
s2_x = 1
s2_ep = 1
s2_v = 0.5
s2_d = 2
n = 1200
np.random.seed(1)
x = np.random.normal(mu_x, np.sqrt(s2_x), n)

burnin = 1000
thinning = 10
posterior_mode = {'beta0':[], 'beta1':[], 'mu_x':[],
                  's2_x':[], 's2_ep':[], 's2_v':[],
                  's2_d':[], 'x': []
                  }
B = 1000

for b in range(B):
    print(f'======================Current Iteration: {b}======================')
    print('\n')
    # Data Generation Part
    v = np.random.normal(0, np.sqrt(s2_v), n)
    # w = np.random.normal(mu_x, np.sqrt(s2_x + s2_v), n)
    # np.random.seed(1)
    d = np.random.normal(0, np.sqrt(s2_d), n)
    # z = np.random.normal(mu_x, np.sqrt(s2_x + s2_d), n)
    # np.random.seed(1)
    # w = np.random.normal(x, np.sqrt(s2_v), n)
    # z = np.random.normal(x, np.sqrt(s2_d), n)
    # np.random.seed(1)
    ep = np.random.normal(0, np.sqrt(s2_ep), n)
    # observed
    w = x+v
    z = x+d
    y = beta0 + beta1*x + ep
    # y = np.random.normal(beta0 + beta1*mu_x, np.sqrt(beta1**2 * s2_x + s2_ep), n)

    act_params = {
        'v':v, 'd':d, 'x':x, 'ep':ep, 'w':w, 'z':z, 'y':y,
        'beta0':beta0, 'beta1':beta1, 'mu_x':mu_x, 's2_x':s2_x,
        's2_ep':s2_ep, 's2_v':s2_v, 's2_d':s2_d
    }
    mod = SLR_ME(y,w,z,act_params,10000,100, thinning=50)
    mod.fit()

    # store posterior mode for each simulation
    params = list(posterior_mode.keys())
    for param in params:
        if param == 'x':
            ax = sns.histplot(mod.params[param][0,:][burnin::thinning], bins=30, kde=True)
        else:
            ax = sns.histplot(mod.params[param][burnin::thinning], bins=30, kde=True)
        plt.close()
        kdeline = ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        mode_idx = np.argmax(ys)
        xs_mode = xs[mode_idx]
        posterior_mode[param].append(xs_mode)

    if b % 10 == 0:
        pickle.dump(posterior_mode, open('../model/MC_SIMUL_SLR_MEM_1200.pkl', 'wb'))
    print('\n')

pickle.dump(posterior_mode, open('../model/MC_SIMUL_SLR_MEM_1200.pkl','wb'))

# mcmc = pickle.load(open('./result.pkl','rb'))
# print(np.mean(mcmc.params['s2_v']))
#
# from statsmodels.graphics.tsaplots import plot_acf
# import matplotlib.pyplot as plt
# plot_acf(np.array(mcmc.params['s2_v']))
