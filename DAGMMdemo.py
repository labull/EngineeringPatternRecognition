import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from EPR.utils import struct, ellipse
from EPR.density_estimation import NIW, mixture


cmap = get_cmap('Dark2')
Ns = 1000
Nt = 1000

K = 4

m = np.array([[-7, 10, - 30, 4], [-5, 18, -20, 8]]).T

S = np.empty((2, 2, K))
S[:, :, 0] = np.array([[5, 0.5], [0.5, 5]])
S[:, :, 1] = np.array([[0.5, 0.3], [0.3, 2]])
S[:, :, 2] = np.array([[2, 0.8], [0.8, 2]])
S[:, :, 3] = np.array([[1, 0.6], [0.6, 1]])

Xs = np.row_stack([np.random.multivariate_normal(m[k], S[:, :, k], Ns)
                   for k in range(K)])
Xt = np.row_stack([np.random.multivariate_normal(m[k], S[:, :, k], Nt)
                   for k in range(K)])

Ys = np.concatenate([(k+1)*np.ones(Ns) for k in range(K)]).astype('int')
Yt = np.concatenate([(k+1)*np.ones(Nt) for k in range(K)]).astype('int')

theta = 20 * (2*np.pi/180)
Ahat = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
Xt = Xt @ Ahat

fig, axs = plt.subplots(1, 2)
[axs[0].scatter(Xs[Ys == k+1, 0], Xs[Ys == k+1, 1], color=cmap(k)) for k in
 range(K)]
[axs[1].scatter(Xt[Yt == k+1, 0], Xt[Yt == k+1, 1], color=cmap(k)) for k in
 range(K)]
plt.show()

##

gmm_prior = struct()
D = 2

# base
gmm_prior.m0 = Xs.mean(0)
gmm_prior.k0 = 5
gmm_prior.S0 = np.cov(Xs.T)
gmm_prior.v0 = D + 5
# mixing
gmm_prior.alpha = 100

# mixture
GMM = mixture(K, NIW, gmm_prior)
GMM.train_supervised(Xs, Ys)
# GMM.EM(Xs)

plt.figure(figsize=[5, 6])
for k in range(K):
    e = ellipse(GMM.base[k].mu_map, GMM.base[k].Sig_map).cov_3
    plt.plot(e[2][:, 0], e[2][:, 1], 'b', lw=.8)    # map cluster
    plt.scatter(Xs[Ys == k+1, 0], Xs[Ys == k+1, 1], color=cmap(k))
plt.show()

##

prior = struct()
prior.alpha = 100
prior.gamma = 1e-2
