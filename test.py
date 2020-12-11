import numpy as np
from simple_linear_reg_v2 import SLR_ME
import pickle

# generate toy data
beta0 = -1
beta1 = 2
gam0 = -1
gam1 = 3
mu_x = 1/2
s2_x = 1
s2_ep = 1
s2_v = 1
s2_d = 1
n = 120

np.random.seed(1)
v = np.random.normal(0, np.sqrt(s2_v), n)
d = np.random.normal(0, np.sqrt(s2_d), n)
x = np.random.normal(mu_x, np.sqrt(s2_x), n)
ep = np.random.normal(0, np.sqrt(s2_ep), n)

# observed
w = x+v
z = gam0 + gam1*x + d
y = beta0 + beta1*x + ep

act_params = {
    'v':v, 'd':d, 'x':x, 'ep':ep, 'w':w, 'z':z, 'y':y,
    'beta0':beta0, 'beta1':beta1, 'gam0':gam0, 'gam1':gam1,
    'mu_x':mu_x, 's2_x':s2_x, 's2_ep':s2_ep, 's2_v':s2_v, 's2_d':s2_d
}
iteration = 50000
mod = SLR_ME(y,w,z,act_params,iteration, 100, thinning=50)
mod.fit()

pickle.dump(mod, open('../model/mem_two_proxies.pkl','wb'))

# mcmc = pickle.load(open('./result.pkl','rb'))
# print(np.mean(mcmc.params['s2_v']))
#
# from statsmodels.graphics.tsaplots import plot_acf
# import matplotlib.pyplot as plt
# plot_acf(np.array(mcmc.params['s2_v']))
