#Mixtures
import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt
import sklearn.mixture as mixture



#cluster_features = ['G','AB','PA','H','1B','2B','3B','HR','R','RBI','BB','IBB','SO','HBP','SF','SH','GDP','SB','CS','AVG','GB','FB','LD','IFFB','Pitches','Balls','Strikes']
#cluster_features = ['1B','2B','3B','HR','R','RBI','SB','CS']
cluster_features = ['SB']

battingStats = '../../playerStats/starter_battingStats.csv'


df = pd.read_csv(battingStats)[cluster_features]

probs = []



k = 3
mod = mixture.BayesianGaussianMixture(n_components = k, max_iter = 100, n_init = 2)
mod.fit(df.values)
#a = mod.score(df.values)
#print(a)
#probs.append(a)


a = mod.predict_proba(df.values)

pops = a.sum(axis = 0) / a.shape[0]

a = mod.predict_proba(df.values)[0]

'''
**************************************************
'''


'''
np.random.seed(12345) # set random seed for reproducibility

k = 3
ndata = 500
spread = 5
centers = np.array([1,3, spread])

# simulate data from mixture distribution
v = np.random.randint(0, k, ndata)
data = centers[v] 

plt.hist(data);


model = pm.Model()

with model:
    # cluster sizes
    p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=k)
    # ensure all clusters have some points
    #p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(p) < .1, -np.inf, 0))


    # cluster centers
    means = pm.Poisson('means', mu=[1, 1, 1],  shape=k)
    # break symmetry
    order_means_potential = pm.Potential('order_means_potential',
                                         tt.switch(means[1]-means[0] < 0, -np.inf, 0)
                                         + tt.switch(means[2]-means[1] < 0, -np.inf, 0))

    # measurement error
    #sd = pm.Uniform('sd', lower=0, upper=20)

    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=ndata)

    # likelihood for each observed value
    points = pm.Poisson('obs',mu=means[category],observed=data)
    print(points)

with model:
    step1 = pm.Metropolis(vars=[p, means])
    step2 = pm.ElemwiseCategorical(vars=[category], values=[0, 1, 2])
    tr = pm.sample(10000, step=[step1, step2])

pm.plots.traceplot(tr, ['p', 'means'])


def cluster_posterior(i=0):
    print('true cluster:', v[i])
    print('  data value:', np.round(data[i],2))
    plt.hist(tr['category'][5000::5,i], bins=[-.5,.5,1.5,2.5,], rwidth=.9)
    plt.axis(xmin=-.5, xmax=2.5)
    plt.xticks([0,1,2])
cluster_posterior()

'''
