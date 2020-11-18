import numpy as np
from simple_linear_reg import SLR_ME
import pickle

# generate toy data
beta0 = -1
beta1 = 1
mu_x = 1/2
s2_x = 1/36
s2_ep = 0.35
s2_v = 1/36
s2_d = 1/100
n = 120

print(f'RR: {(s2_x)/(s2_x + s2_v)}')

np.random.seed(1)
v = np.random.normal(0, np.sqrt(s2_v), n)
# w = np.random.normal(mu_x, np.sqrt(s2_x + s2_v), n)
# np.random.seed(1)
d = np.random.normal(0, np.sqrt(s2_d), n)
# z = np.random.normal(mu_x, np.sqrt(s2_x + s2_d), n)
# np.random.seed(1)
x = np.random.normal(mu_x, np.sqrt(s2_x), n)
# w = np.random.normal(x, np.sqrt(s2_v), n)
# z = np.random.normal(x, np.sqrt(s2_d), n)
# np.random.seed(1)
ep = np.random.normal(0, np.sqrt(s2_ep), n)
# observed
w = x+v
z = x+d
y = beta0 + beta1*x + ep
# y = np.random.normal(beta0 + beta1*mu_x, np.sqrt(beta1**2 * s2_x + s2_ep), n)

print(x[:5])
print(y[:5])
act_params = {
    'v':v, 'd':d, 'x':x, 'ep':ep, 'w':w, 'z':z, 'y':y,
    'beta0':beta0, 'beta1':beta1, 'mu_x':mu_x, 's2_x':s2_x,
    's2_ep':s2_ep, 's2_v':s2_v, 's2_d':s2_d
}
mod = SLR_ME(y,w,z,act_params,10000,100, thinning=50)
mod.fit()

pickle.dump(mod, open('../model/mem_result_with_s2_v_d_120_simul.pkl','wb'))

# mcmc = pickle.load(open('./result.pkl','rb'))
# print(np.mean(mcmc.params['s2_v']))
#
# from statsmodels.graphics.tsaplots import plot_acf
# import matplotlib.pyplot as plt
# plot_acf(np.array(mcmc.params['s2_v']))
