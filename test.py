import numpy as np
from simple_linear_reg import SLR_ME
import pickle

# generate toy data
beta0 = -1
beta1 = 1
mu_x = 1/2
s2_x = 1/36
s2_ep = 0.35
s2_v = 1/30
n = 1000

print(f'RR: {(s2_x)/(s2_x + s2_v)}')

np.random.seed(1)
v = np.random.normal(0, np.sqrt(s2_v), n)
x = np.random.normal(mu_x, np.sqrt(s2_x), n)
ep = np.random.normal(0, np.sqrt(s2_ep), n)
# observed
w = x+v
y = beta0 + beta1*x + ep

mod = SLR_ME(y,w,100000,100, thinning=50)
mod.fit()

pickle.dump(mod, open('../model/mem_result_with_s2_v.pkl','wb'))

# mcmc = pickle.load(open('./result.pkl','rb'))
# print(np.mean(mcmc.params['s2_v']))
#
# from statsmodels.graphics.tsaplots import plot_acf
# import matplotlib.pyplot as plt
# plot_acf(np.array(mcmc.params['s2_v']))
