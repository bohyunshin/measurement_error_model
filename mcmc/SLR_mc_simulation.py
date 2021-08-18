import numpy as np
from simple_linear_reg_basic import SLR_BASIC
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

# generate toy data
beta0 = -1
beta1 = 1
s2_ep = 1
n = 1000
burnin = 1000
thinning = 10
posterior_mode = {'beta0':[], 'beta1':[], 's2_ep':[]}
B = 300

np.random.seed(1224)
for b in range(B):
    print(f'======================Current Iteration: {b}======================')
    print('\n')


    ep = np.random.normal(0, np.sqrt(s2_ep), n)
    # observed
    x = np.random.uniform(10, 40, n)
    y = beta0 + beta1*x + ep
    act_params = {
        'ep':ep,  'y':y,
        'beta0':beta0, 'beta1':beta1,
        's2_ep':s2_ep
    }

    mod = SLR_BASIC(y,x,act_params,10000,100, thinning=50)
    mod.fit()

    # store posterior mode for each simulation
    params = ['beta0','beta1','s2_ep']
    for param in params:
        ax = sns.histplot(mod.params[param][burnin::thinning], bins=30, kde=True)
        plt.close()
        kdeline = ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        mode_idx = np.argmax(ys)
        xs_mode = xs[mode_idx]
        posterior_mode[param].append(xs_mode)

    if b % 10 == 0:
        pickle.dump(posterior_mode, open('../model_mc_simul/MC_SIMUL_SLR_1000.pkl', 'wb'))

    print('\n')




