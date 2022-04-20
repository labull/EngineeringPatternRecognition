import numpy as np
from scipy.sparse.linalg import eigs
from scipy.stats import multivariate_normal as mvn
from scipy.special import digamma

from EPR.utils import chinv


class DAGMM:
    # domain-adapted Gaussian mixture model (DA-GMM)
    def __init__(self, sGMM, prior):
        # source (Xs) GMM
        self.K = sGMM.K  # no. components
        self.D = sGMM.base[0].Sig_map.shape[0]  # dims of the source domain
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
        self.gamma = prior.gamma
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
            r_l = np.zeros((Y_l.size, self.K))    # responsibility 4 labelled
            # data
            for ii in range(Y_l.size):
                r_l[ii, Y_l[ii] - 1] = 1     # dirac labels

        N, p = Phi.shape  # dims from the kernel

        # -- the A prior: matrix-normal MN(A | M, V, K)
        # projection matrix:    A = D x p
        # row variance:         V = D x D
        # column variance:      K = p x p

        # row variance
        Kcov = self.gamma * Phi.T @ Phi
        L_K = np.linalg.cholesky(Kcov)
        LK_inv = np.linalg.inv(L_K)
        Kinv = LK_inv.T @ LK_inv

        # init A matrix
        # (one) sample from prior option
        Acol = mvn.rvs(np.ones(self.D * p), np.kron(Kinv, self.V))
        self.A = Acol.reshape(self.D, p, order='F')

        for _ in range(itt):

            # expectation of quadratic form wrt A
            Qk = np.linalg.solve(L_K, Phi.T)
            phi_Kinv_phi = np.sum(Qk * Qk, 0)
            E_A = [[]] * self.K
            for k in range(self.K):
                Q = self.L_lam[k].T @ (Phi @ self.A.T - self.mu[k]).T
                E_A[k] = (phi_Kinv_phi * np.trace(self.V @ self.lam[k].T)
                          + np.sum(Q * Q, 0))

            # expectation of log(pi) wrt dirichlet
            E_lnPi = digamma(self.alpha) - digamma(self.alpha.sum())

            # rho variable
            lnRho = (- 0.5 * self.D * np.log(2 * np.pi) + self.log_det_lam +
                     E_lnPi - 0.5 * np.column_stack(E_A))
            ln_r = (lnRho.T - np.log(np.sum(np.exp(lnRho.T - lnRho.max(1)), 0))
                    - lnRho.max(1)).T
            res = np.exp(ln_r)  # responsibilities

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
            for k in range(self.K):
                Syx[k] = np.outer(self.mu[k], (res[:, k] @ Phi))
                S[k] = (res[:, k][:, np.newaxis] * Phi).T @ Phi

            S = sum(S)
            # TODO: make the prior a general M (rather than ones)
            Syx = sum(Syx) + np.ones((self.D, p)) @ Kcov
            Sxx = S + Kcov

            self.A = Syx @ chinv(Sxx)
            L_K = np.linalg.cholesky(Sxx)   # update L_k for E_A

            self.H = Phi @ self.A.T

            # TODO: calculate the lower bound (to check convergence)

        self.Z = res

    def project(self, Phi_test):
        # project test (target) data onto the source domain
        return Phi_test @ self.A.T

    # TODO: methods - sample A matrix


class TCA:
    # an implementation of TCA domain adaptation
    def __init__(self, source_data, target_data, kernel):
        Xs = source_data
        Xt = target_data
        self.X = np.row_stack((Xs, Xt))  # data from each domain combined

        # normalise
        self.kernel = kernel
        self.K = kernel(self.X, self.X)  # combined data through kernel
        self.ns = Xs.shape[0]  # number of source data
        self.nt = Xt.shape[0]  # number of target data
        self.n = self.ns + self.nt  # no. points combined
        self.d = self.X.shape[1]  # dimensionality
        self.L = np.ones((self.n, self.n)) * -1 / (self.ns * self.nt)
        self.L[:self.ns, :self.ns] = 1 / self.ns ** 2
        self.L[self.ns:, self.ns:] = 1 / self.nt ** 2

        # calculate H (centering matrix)
        self.H = np.eye(self.n) - np.ones((self.n, self.n)) / self.n

    def solve(self, dim, mu=0):
        # mu - trade-off parameter
        # dim - no of eigenvectors / dimension of the embedding    
        J = (np.linalg.inv(mu * np.eye(self.n) + self.K @ self.L @ self.K)
             @ self.K @ self.H @ self.K)

        _, vecs = eigs(J, k=dim)
        self.W = vecs.real

        self.Z = (self.W.T @ self.K.T).T  # transform

        # # normalise Z space
        # self.Z = ((self.Z - self.Z.mean(axis=0))
        #           /self.Z.std(axis=0)) # normalise

        self.Zs = self.Z[:self.ns, :]  # source part
        self.Zt = self.Z[self.ns:, :]  # target part

    def test(self, source_test_data, target_test_data):
        Xs_test = source_test_data
        Xt_test = target_test_data
        self.X_test = np.row_stack((Xs_test, Xt_test))
        ns_test = Xs_test.shape[0]  # number of source data
        Kxxt = self.kernel(self.X, self.X_test)
        self.Z_test = Kxxt.T @ self.W

        # # normalise Z space
        # self.Z_test = ((self.Z_test - self.Z_test.mean(axis=0))
        #                 /self.Z_test.std(axis=0)) # normalise

        self.Zs_test = self.Z_test[:ns_test, :]  # source part
        self.Zt_test = self.Z_test[ns_test:, :]  # target part
