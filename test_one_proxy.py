import numpy as np
from simple_linear_reg_v3_jeffrey_prior import SLR_ME_ONE_PROXY
import pickle

# generate toy data
beta0 = -1
beta1 = 2

mu_x = 1/2
s2_x = 1
s2_ep = 1
s2_v = 1

n = 120

np.random.seed(1)
v = np.random.normal(0, np.sqrt(s2_v), n)
x = np.random.normal(mu_x, np.sqrt(s2_x), n)
ep = np.random.normal(0, np.sqrt(s2_ep), n)
# observed
w = x+v
y = beta0 + beta1*x + ep
# y = np.random.normal(beta0 + beta1*mu_x, np.sqrt(beta1**2 * s2_x + s2_ep), n)

print(x[:5])
print(y[:5])
act_params = {
    'v':v, 'x':x, 'ep':ep, 'w':w, 'y':y,
    'beta0':beta0, 'beta1':beta1,
    'mu_x':mu_x, 's2_x':s2_x, 's2_ep':s2_ep, 's2_v':s2_v
}
mod = SLR_ME_ONE_PROXY(y,w,act_params,50000,100, thinning=50)
mod.fit()

pickle.dump(mod, open('../model/mem_result_with_jeffrey.pkl','wb'))

# mcmc = pickle.load(open('./result.pkl','rb'))
# print(np.mean(mcmc.params['s2_v']))
#
# from statsmodels.graphics.tsaplots import plot_acf
# import matplotlib.pyplot as plt
# plot_acf(np.array(mcmc.params['s2_v']))
