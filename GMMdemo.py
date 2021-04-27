import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wishart, entropy
from scipy.stats import multivariate_normal as normal

# -- EPR 
from EPR.utils import struct, ellipse
from EPR.density_estimation import NIW, mixture

#%% base NIW
prior = struct()
D = 2
prior.m0 = np.zeros(D)
prior.k0 = 1
prior.S0 = np.eye(D)
prior.v0 = D

# sample data
Cov = wishart.rvs(D, np.eye(D), 1, random_state=2)
mean = [1,1]
X = normal.rvs(mean, Cov, size=100, random_state=2)
plt.figure(figsize=[5,6], dpi=300)
plt.scatter(X[:,0], X[:,1], s=2)

# NIW
C = NIW(prior)

# # test for the M-step (weighted MAP params with responsibility)
# C.M_step(X, np.ones(100))
# standard training
C.train(X)
# sample from the predictive
samp = C.x_sample(int(5e2))

# plot ellipses
e1 = ellipse(C.mu_map, C.Sig_map).cov_3
e2 = ellipse(mean, Cov).cov_3
plt.plot(e1[2][:,0], e1[2][:,1], 'b') # map estimate
plt.plot(e2[2][:,0], e2[2][:,1], 'k') # underlying ground truth

# predict mesh grid
xx, yy = np.meshgrid(np.linspace(X.min(0)[0]-2,X.max(0)[0]+2,100), 
                     np.linspace(X.min(0)[1]-2,X.max(0)[1]+2,100))
Xt = np.column_stack([xx.ravel(), yy.ravel()]) # test data
lp = C.predict(Xt)
LP = np.array(lp).reshape(xx.shape)

e3 = ellipse(C.post_pred.mu, C.post_pred.S).cov_3
plt.plot(e3[2][:,0], e2[2][:,1], 'r') # posterior predictive
c = plt.contourf(xx, yy, np.exp(LP), alpha=.7, levels=50, cmap='GnBu_r')
plt.scatter(samp[:,0], samp[:,1], c='k', s=.1, alpha=.8) # samples
plt.scatter(X[:,0], X[:,1], s=1, alpha=.8) # training-data
plt.colorbar(c)

#%% --- make multi-class data
K = 3 # number of clusters
m = [[2,5], [0, -3], [-7,-4]] # means
# training data
X = []
Y = []
# unlabelled training data (for semi-supervised examples)
Xu = []

# number of labelled/unlabelled obvs
nl = 50
nu = 500

et = []

plt.figure(figsize=[4,6], dpi=300)
# sample data
for k in range(K):
    C = wishart.rvs(5, np.eye(D), 1)
    x = normal.rvs(m[k], C, size=nl)
    xu = normal.rvs(m[k], C, size=nu)
    
    plt.scatter(x[:,0], x[:,1], s=2)
    e = ellipse(m[k], C).cov_3
    plt.plot(e[2][:,0], e[2][:,1], 'k', lw=.8) # underlying ground truth
    et.append(e)
    
    X.append(x)
    Xu.append(xu)
    Y.append(np.tile(k+1, x.shape[0]))

# labelled data
X = np.row_stack(X)
Y = np.concatenate(Y)
# unlabelled
Xu = np.row_stack(Xu)


#%% --- supervised mixture model

prior.alpha = 1
GMM = mixture(K, NIW, prior)
GMM.train_supervised(X, Y)

plt.figure(figsize=[5,6], dpi=300)
for k  in range(K):
    e = ellipse(GMM.base[k].mu_map, GMM.base[k].Sig_map).cov_3
    plt.plot(e[2][:,0], e[2][:,1], 'b', lw=.8) # map cluster
    plt.plot(et[k][2][:,0], et[k][2][:,1], 'k', lw=.8) # ground truth

# mesh grid
xx, yy = np.meshgrid(np.linspace(X.min(0)[0]-5,X.max(0)[0]+5,100), 
                     np.linspace(X.min(0)[1]-5,X.max(0)[1]+5,100))
Xt = np.column_stack([xx.ravel(), yy.ravel()]) # test data

py_xt = GMM.predict(Xt)
lpxt = GMM.lpx.reshape(xx.shape)
c = plt.contourf(xx, yy, lpxt, alpha=.9, levels=10, cmap='GnBu_r')
plt.scatter(X[:,0], X[:,1], s=1)
plt.colorbar(c)
plt.title('supervised')

#%% --- unsupervised

GMM = mixture(K, NIW, prior)
GMM.EM(X)

plt.figure(figsize=[5,6], dpi=300)
for k  in range(K):
    e = ellipse(GMM.base[k].mu_map, GMM.base[k].Sig_map).cov_3
    plt.plot(e[2][:,0], e[2][:,1], 'r', lw=.8) # map cluster
    plt.plot(et[k][2][:,0], et[k][2][:,1], 'k', lw=.8) # ground truth

py_xt = GMM.predict(Xt)
lpxt = GMM.lpx.reshape(xx.shape)
c = plt.contourf(xx, yy, lpxt, alpha=.9, levels=10, cmap='GnBu_r')
plt.scatter(X[:,0], X[:,1], s=1)
plt.colorbar(c)
plt.title('unsupervised')
    
# plot lml
plt.figure(figsize=[5,6], dpi=300)
plt.xlabel('iteration')
plt.ylabel('log-marginal-likelihood')
plt.plot(GMM.lml)

#%% --- semi-supervised

GMM = mixture(K, NIW, prior)
GMM.semisupervisedEM(X, Y, Xu)

plt.figure(figsize=[5,6], dpi=300)
# ellipses
for k  in range(K):
    e = ellipse(GMM.base[k].mu_map, GMM.base[k].Sig_map).cov_3
    plt.plot(e[2][:,0], e[2][:,1], 'r', lw=.8) # map cluster
    plt.plot(et[k][2][:,0], et[k][2][:,1], 'k', lw=.8) # ground truth
    
py_xt = GMM.predict(Xt)
lpxt = GMM.lpx.reshape(xx.shape)
c = plt.contourf(xx, yy, lpxt, alpha=.9, levels=10, cmap='GnBu_r')
plt.scatter(X[:,0], X[:,1], c=Y, s=1)
plt.scatter(Xu[:,0], Xu[:,1], s=1, c='k', alpha=.6)
plt.colorbar(c)
plt.title('semi-supervised')
    
# plot joint-ll
plt.figure(figsize=[5,6], dpi=300)
plt.xlabel('iteration')
plt.ylabel('log-marginal-likelihood')
plt.plot(GMM.ll)

#%% --- uncertainty sampling (for active learning)

# entropy queries of unlabelled data
nq = 50 # number of queries
Hy = entropy(GMM.r_ul, axis=1) # shanon-entropy

hi = Hy.argsort()[-nq:] # idxs with highest entropy
li = GMM.lpx_ul.argsort()[:nq] # idxs with low-likelihood

plt.figure(figsize=[4,6], dpi=300)
# ellipses
for k  in range(K):
    e = ellipse(GMM.base[k].mu_map, GMM.base[k].Sig_map).cov_3
    plt.plot(e[2][:,0], e[2][:,1], 'k', lw=.2) # map cluster


plt.scatter(Xu[:,0], Xu[:,1], s=1, c='k', alpha=.3)
# high entropy points (larger blue blobs)
plt.scatter(Xu[hi,0], Xu[hi,1], marker='o', 
            s=np.exp(5*Hy[hi]), c='b', alpha=.3)
# low-likelihood queries (larger green blobs)
plt.scatter(Xu[li,0], Xu[li,1], marker='o', 
            s=100-5e4*np.exp(GMM.lpx_ul[li]), c='g', alpha=.3)
plt.title('uncertainty sampling')
