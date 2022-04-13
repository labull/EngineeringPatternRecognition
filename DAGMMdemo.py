import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.stats import multinomial
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal as mvn
from scipy.special import digamma
#
from EPR.utils import struct, ellipse, chinv
from EPR.density_estimation import NIW, mixture

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
                   for k in range(K-1)])  # one less class in target

Ys = np.concatenate([(k + 1) * np.ones(Ns) for k in range(K)]).astype('int')
# Yt = np.concatenate([(k + 1) * np.ones(Nt) for k in range(K)]).astype('int')
# one less class in target
Yt = np.concatenate([(k + 1) * np.ones(Nt) for k in range(K-1)]).astype('int')

theta = 40 * (2 * np.pi / 180)
Ahat = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
Xt = Xt @ Ahat

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
[axs[0].scatter(Xs[Ys == k + 1, 0], Xs[Ys == k + 1, 1], color=cmap(k)) for k in
 range(K)]
[axs[1].scatter(Xt[Yt == k + 1, 0], Xt[Yt == k + 1, 1], color=cmap(k)) for k in
 range(K)]
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

plt.figure(figsize=[5, 5])
for k in range(K):
    e = ellipse(sGMM.base[k].mu_map, sGMM.base[k].Sig_map).cov_3
    plt.plot(e[2][:, 0], e[2][:, 1], 'b', lw=.8)  # map cluster
    plt.scatter(Xs[Ys == k + 1, 0], Xs[Ys == k + 1, 1], color=cmap(k))
plt.tight_layout()
plt.show()

##

class DAGMM:
    # domain-adapted Gaussian mixture model (DA-GMM)
    def __init__(self, sGMM, prior):
        # source (Xs) GMM
        self.K = sGMM.K  # no. components
        self.mu = [sGMM.base[k].mu_map for k in range(self.K)]  # locs
        # source covariance (Vs) and precision (lam) matrices per base in GMM
        self.Vs = [sGMM.base[k].Sig_map for k in range(self.K)]  # lam_inv
        Linv_sig = [np.linalg.inv(np.linalg.cholesky(V)) for V in self.Vs]
        self.lam = [Linv_sig[k].T @ Linv_sig[k] for k in range(self.K)]
        self.L_lam = [np.linalg.cholesky(self.lam[k]) for k in range(self.K)]
        self.log_det_lam = [np.sum(np.log(np.diag(self.L_lam[k])))
                            for k in range(self.K)]  # 1/2*log[det[lam]]

        # projection
        self.alpha0 = prior.alpha
        self.alpha = np.ones(self.K) * prior.alpha  # init Dir posterior
        self.A = None  # init regression weights
        # row variance (output) as the average of the sGMM conditionals
        self.V = np.mean(self.Vs, 0)
        self.H = None   # projected target data
        self.Z = None   # predicted labels

    def train(self, Phi, Phi_l=None, Y_l=None, itt=10):
        # -- target data (Phi(Xt))

        # if labelled data, stack with unlabelled
        if Phi_l is not None:
            Phi = np.row_stack((Phi, Phi_l))  # stack labelled / unlabelled Phi
            r_l = np.zeros((Y_l.size, K))    # responsibility 4 labelled data
            for ii in range(Y_l.size):
                r_l[ii, Y_l[ii] - 1] = 1     # dirac labels

        N, p = Phi.shape  # dims from the kernel
        D = sGMM.base[0].Sig_map.shape[0]  # dims of the source domain

        # -- the A prior: matrix-normal MN(A | M, V, K)
        # projection matrix:    A = D x p
        # row variance:         V = D x D
        # column variance:      K = p x p

        # row variance
        Kcov = prior.gamma * Phi.T @ Phi
        L_K = np.linalg.cholesky(Kcov)
        LK_inv = np.linalg.inv(L_K)
        Kinv = LK_inv.T @ LK_inv

        # Kinv, L_k, = chinv(Kcov)

        # init A matrix
        # (one) sample from prior option
        Acol = mvn.rvs(np.ones(D * p), np.kron(Kinv, self.V))
        self.A = Acol.reshape(D, p, order='F')
        # # near the ground truth
        # self.A = Ahat + np.random.normal(size=Ahat.shape)*.1

        for _ in range(itt):

            fig = plt.figure(figsize=[5, 5])
            self.H = Phi @ self.A.T
            for k in range(K):
                e = ellipse(sGMM.base[k].mu_map, sGMM.base[k].Sig_map).cov_3
                plt.plot(e[2][:, 0], e[2][:, 1], 'k', lw=.8)  # map cluster
                plt.scatter(Xs[Ys == k + 1, 0], Xs[Ys == k + 1, 1],
                            color='k', s=0.5, alpha=0.1)
                if Phi_l is None:
                    plt.scatter(self.H[Yt == k + 1, 0], self.H[Yt == k + 1, 1],
                                color=cmap(k), zorder=0)
                else:   # when semi-supervised use shuffled plotting labels
                    plt.scatter(self.H[Y_t == k + 1, 0],
                                self.H[Y_t == k + 1, 1],
                                color=cmap(k), zorder=0)

            plt.tight_layout()
            plt.show(block=False)
            plt.close()

            # expectation of quadratic form wrt A
            Qk = np.linalg.solve(L_K, Phi.T)
            phi_Kinv_phi = np.sum(Qk * Qk, 0)
            E_A = [[]] * K
            for k in range(K):
                Q = self.L_lam[k].T @ (Phi @ self.A.T - self.mu[k]).T
                E_A[k] = (phi_Kinv_phi * np.trace(self.V @ self.lam[k].T)
                          + np.sum(Q * Q, 0))

            # expectation of log(pi) wrt dirichlet
            E_lnPi = digamma(self.alpha) - digamma(self.alpha.sum())

            # rho variable
            lnRho = (- 0.5 * D * np.log(2 * np.pi) + self.log_det_lam +
                     E_lnPi - 0.5 * np.column_stack(E_A))
            ln_r = (lnRho.T - np.log(np.sum(np.exp(lnRho.T - lnRho.max(1)), 0))
                    - lnRho.max(1)).T
            res = np.exp(ln_r)  # responsibilities
            # res = resp_gt

            # if labelled data (semi-supervised) update responsibilities
            if Phi_l is not None:
                res[-Y_l.size:] = r_l

            Nk = res.sum(0)

            # variational maximisation step
            # q(pi)
            self.alpha = self.alpha0 + Nk
            # q(A)
            Syx = [[]] * self.K
            S = [[]] * self.K
            for k in range(K):
                Syx[k] = np.outer(self.mu[k], (res[:, k] @ Phi))
                S[k] = (res[:, k][:, np.newaxis] * Phi).T @ Phi

            S = sum(S)
            # TODO: make the prior a general M (rather than ones)
            Syx = sum(Syx) + np.ones((D, p)) @ Kcov
            Sxx = S + Kcov

            self.A = Syx @ chinv(Sxx)
            L_K = np.linalg.cholesky(Sxx)   # update L_k for E_A

            self.H = Phi @ self.A.T

        self.Z = res

##

prior = struct()
prior.alpha = 1
prior.gamma = 1e-3

dagmm = DAGMM(sGMM, prior)

# ground truth responsibilities
resp_gt = np.zeros((Yt.size, K))
for ii, resp in enumerate(resp_gt):
    resp_gt[ii, Yt[ii] - 1] = 1

##

# -- unsupervised
# design matrix
Phi = np.column_stack([Xt])
# train supervised
dagmm.train(Phi, itt=5)

# -- semi-supervised split into labelled and unlabelled
# Phi = np.column_stack([Xt])
Phi, Phi_l, Y_ul, Y_l = train_test_split(Phi, Yt, stratify=Yt, test_size=0.5)
Y_t = np.append(Y_ul, Y_l)      # for plotting
# train
dagmm.train(Phi, Phi_l, Y_l, itt=5)


print(Ahat)
print(dagmm.A)

##

