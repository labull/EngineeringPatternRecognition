import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.model_selection import train_test_split
#
from EPR.utils import struct, ellipse
from EPR.density_estimation import NIW, mixture
from EPR.transformations import DAGMM

cmap = get_cmap('Dark2')

Ns = 1000
Nt = 1000

K = 4

m = np.array([[-7, 10, - 30, 4], [-5, 18, -20, 8]]).T

S = [[]] * K
S[0] = np.array([[5, 0.5], [0.5, 5]])
S[1] = np.array([[0.5, 0.3], [0.3, 2]])
S[2] = np.array([[2, 0.8], [0.8, 2]])
S[3] = np.array([[1, 0.6], [0.6, 1]])

Xs = np.row_stack([np.random.multivariate_normal(m[k], S[k], Ns)
                   for k in range(K)])
# Xt = np.row_stack([np.random.multivariate_normal(m[k], S[k], Nt)
#                    for k in range(K)])
Xt = np.row_stack([np.random.multivariate_normal(m[k], S[k], Nt)
                   for k in range(K-1)])  # ** one less class in target

Ys = np.concatenate([(k + 1) * np.ones(Ns)
                     for k in range(K)]).astype('int')
# Yt_all = np.concatenate([(k + 1) * np.ones(Nt)
#                          for k in range(K)]).astype('int')
# ** one less class in target
Yt_all = np.concatenate([(k + 1) * np.ones(Nt)
                         for k in range(K-1)]).astype('int')

# rotate target
theta = 40 * (2 * np.pi / 180)
Ahat = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
Xt = Xt @ Ahat.T
Phi_all = np.column_stack([Xt])       # design matrix

# including shift
# Ahat = np.column_stack((50*np.ones(2), Ahat))
# Phi_all = np.column_stack([np.ones(Xt.shape[0]), Xt])       # design matrix
# Xt = Phi_all @ Ahat.T


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
[axs[0].scatter(Xs[Ys == k + 1, 0], Xs[Ys == k + 1, 1], color=cmap(k)) for k in
 range(K)]
axs[0].set_title(r'$D_s = \{X_s, Y_s\}$')
[axs[1].scatter(Xt[Yt_all == k + 1, 0], Xt[Yt_all == k + 1, 1],
                color=cmap(k), alpha=.3) for k in range(K)]
axs[1].set_title(r'$D_t = \{X_t\}$')
plt.tight_layout()
plt.show()

##

gmm_prior = struct()
D = 2

# base
gmm_prior.m0 = Xs.mean(0)
gmm_prior.k0 = 1
gmm_prior.S0 = np.cov(Xs.T)
gmm_prior.v0 = D + 5
# mixing
gmm_prior.alpha = 10

# mixture
sGMM = mixture(K, NIW, gmm_prior)
sGMM.train_supervised(Xs, Ys)
# sGMM.EM(Xs)       # LABEL SWITCHING ISSUES IF SEMI SUPERVISED DA


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
[axs[0].scatter(Xs[Ys == k + 1, 0], Xs[Ys == k + 1, 1], color=cmap(k)) for k in
 range(K)]
axs[0].set_title(r'$D_s = \{X_s, Y_s\}$')
[axs[1].scatter(Xt[Yt_all == k + 1, 0], Xt[Yt_all == k + 1, 1],
                color=cmap(k), alpha=.3) for k in range(K)]
axs[1].set_title(r'$D_t = \{X_t\}$')

for k in range(K):
    e = ellipse(sGMM.base[k].mu_map, sGMM.base[k].Sig_map).cov_3
    axs[0].plot(e[2][:, 0], e[2][:, 1], 'k', lw=.8)  # map cluster
plt.tight_layout()
plt.show()

##
# projection priors

prior = struct()
prior.alpha = 1
prior.gamma = 1e-3

##

Yt = Yt_all.copy()

# test train split
Phi, Phi_test, Yt, Yt_test = train_test_split(Phi_all, Yt_all, stratify=Yt_all,
                                              test_size=0.66)
# -- unsupervised
# train supervised
dagmm = DAGMM(sGMM, prior)
dagmm.train(Phi, itt=10)

# plot
fig = plt.figure(figsize=[5, 5])
for k in range(K):
    e = ellipse(sGMM.base[k].mu_map, sGMM.base[k].Sig_map).cov_3
    plt.plot(e[2][:, 0], e[2][:, 1], 'k', lw=.8)  # map cluster
    plt.scatter(Xs[Ys == k + 1, 0], Xs[Ys == k + 1, 1],
                color='k', s=0.5, alpha=0.1)
    plt.scatter(dagmm.H[Yt == k + 1, 0], dagmm.H[Yt == k + 1, 1],
                color=cmap(k), zorder=0)
plt.show()

# predict
z_test = sGMM.predict(dagmm.project(Phi_test))

acc_us = 100 * (np.sum(((np.argmax(z_test, 1) + 1) == Yt_test)) /
                Yt_test.size)
print(acc_us)
print(sum(sGMM.lpx))

##
# -- semi-supervised split into labelled and unlabelled

Phi_ul, Phi_l, Y_ul, Y_l = train_test_split(Phi, Yt, stratify=Yt,
                                            test_size=.5)
Y_t = np.append(Y_ul, Y_l)      # for plotting

# train
dagmm = DAGMM(sGMM, prior)
dagmm.train(Phi_ul, Phi_l, Y_l, itt=10)

# plot
fig = plt.figure(figsize=[5, 5])
for k in range(K):
    e = ellipse(sGMM.base[k].mu_map, sGMM.base[k].Sig_map).cov_3
    plt.plot(e[2][:, 0], e[2][:, 1], 'k', lw=.8)  # map cluster
    plt.scatter(Xs[Ys == k + 1, 0], Xs[Ys == k + 1, 1],
                color='k', s=0.5, alpha=0.1)
    # when semi-supervised use shuffled plotting labels
    plt.scatter(dagmm.H[Y_t == k + 1, 0],
                dagmm.H[Y_t == k + 1, 1],
                color=cmap(k), zorder=0)
plt.show()

# predict
z_test = sGMM.predict(dagmm.project(Phi_test))
acc_ss = 100 * (np.sum(((np.argmax(z_test, 1) + 1) == Yt_test)) /
                Yt_test.size)
print(acc_ss)
print(sum(sGMM.lpx))
