import numpy as np
from simple_linear_reg_basic import SLR_BASIC
import pickle

# generate toy data
beta0 = -10
beta1 = 5
s2_ep = 10
n = 120

np.random.seed(1)
ep = np.random.normal(0, np.sqrt(s2_ep), n)
# observed
x = np.random.uniform(10, 40, 120)
y = beta0 + beta1*x + ep

print(x[:5])
print(y[:5])
act_params = {
    'ep':ep,  'y':y,
    'beta0':beta0, 'beta1':beta1,
    's2_ep':s2_ep
}
mod = SLR_BASIC(y,x,act_params,10000,100, thinning=50)
mod.fit()

pickle.dump(mod, open('../model/SLR_basic.pkl','wb'))

# mcmc = pickle.load(open('./result.pkl','rb'))
# print(np.mean(mcmc.params['s2_v']))
#
# from statsmodels.graphics.tsaplots import plot_acf
# import matplotlib.pyplot as plt
# plot_acf(np.array(mcmc.params['s2_v']))
