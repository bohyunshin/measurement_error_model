import numpy as np
from simple_linear_reg_basic_uniform_prior import SLR_BASIC
import pickle

# generate toy data
beta0 = -1
beta1 = 1
s2_ep = 1
n = 120


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

pickle.dump(mod, open('../model/SLR_basic_120.pkl','wb'))

# mcmc = pickle.load(open('./result.pkl','rb'))
# print(np.mean(mcmc.params['s2_v']))
#
# from statsmodels.graphics.tsaplots import plot_acf
# import matplotlib.pyplot as plt
# plot_acf(np.array(mcmc.params['s2_v']))
