# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:59:36 2020

@author: lucas
"""

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import emcee
import corner


def log_priorL(theta):
    mu, s=theta
    if -20.0<mu<20.0 and 0.0<s<100.0:
        return 0.0
    return -np.inf
def log_likelihoodL(theta, x):
    #logl = -np.sum(theta) + np.sum(n*np.log(theta))-np.sum(math.lgamma(n + 1))
    mu, s=theta
    logl = np.sum((mu-x)/s-np.log(s)-2*np.log(1+np.exp((mu-x)/s)))
    return logl
def log_posteriorL(theta, x):
    lp=log_priorL(theta)
    logl = log_likelihoodL(theta, x)
    logpos = logl + lp
    return logpos
####################################################################################
def log_priorP(theta):
    if -10<theta<10:
        return 0.0
    return -np.inf
def log_likelihoodP(theta, x):
    logl = np.sum(x*np.log(theta)-theta-np.log(scipy.special.gamma(x+1)))
    return logl
def log_posteriorP(theta, x):
    lp=log_priorP(theta)
    logl = log_likelihoodP(theta, x)
    logpos = logl + lp
    return logpos


#LETTURA DEI DATI
data = pd.read_csv("dpc-covid19-ita-andamento-nazionale.csv")
n=np.asarray(data.nuovi_positivi/(24*60))
new=np.array_split(n,2)
n1=new[0]
n2=new[1]
datanew=np.array_split(data.data,2)
data1=datanew[0]
data2=datanew[1]


#PLOT DEI NUOVI POSITIVI
fig, axes = plt.subplots(nrows=3,figsize=(10,7))
plt.sca(axes[0])
plt.xticks([7,37,67,98,128,159,190,220,251,281], ['01/03', '01/04', '01/05', '01/06', '01/07', '01/08', '01/09','01/10','01/11','01/12'])
axes[0].plot(data.data,n)

plt.sca(axes[1])
plt.xticks([7,37,67,98,128], ['01/03', '01/04', '01/05', '01/06', '01/07'])
axes[1].plot(data1,n1)

plt.sca(axes[2])
plt.xticks([14,45,75,106,136], ['01/08', '01/09','01/10','01/11','01/12'])
axes[2].plot(data2,n2)


#EMCEE FIT DISTRIBUZIONE LOGISTICA
pos = [2.7,3.3] + 1e-2*np.random.randn(16,2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posteriorL, args=[n]) #Seleziona il set di dati:n, n1 o n2
sampler.run_mcmc(pos, 3000);

fig, axes = plt.subplots(2, figsize=(10,7), sharex=True)
samples = sampler.get_chain()
print(samples.shape)
labels=['$\mu$','s']
for i in range(ndim):
    ax=axes[i]
    ax.plot(samples[:,:,i])
    ax.set_ylabel(labels[i])
    ax.set_xlim(0,len(samples))
tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=100, flat=True)
flat_blob = sampler.get_blobs(flat=True, discard=100)

fig = corner.corner(flat_samples, labels=labels)



#EMCEE FIT DISTRIBUZIONE POISSONIANA
pos = [4.3] + 1e-2*np.random.randn(16,1)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posteriorP, args=[n]) #Seleziona il set di dati:n, n1 o n2
sampler.run_mcmc(pos, 3000);


samples = sampler.get_chain()
print(samples.shape)
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(samples[:,:,0])
ax.set_ylabel('$\lambda$')
ax.set_xlim(0,len(samples))
tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=100, flat=True)
flat_blob = sampler.get_blobs(flat=True, discard=100)
labels=['$\lambda$']
fig = corner.corner(flat_samples,labels=labels)



