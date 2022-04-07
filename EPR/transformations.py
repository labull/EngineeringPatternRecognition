import numpy as np
from scipy.sparse.linalg import eigs


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


class DAGMM:
    # domain-adapted Gaussian mixture model (DAGMM)
    def __init__(self, GMM, prior):
        # no. components
        self.K = GMM.K
        # locs
        self.mu = [GMM.base[k].mu_map for k in range(self.K)]
        # precision matrices
        Linv_sig = [np.linalg.inv(np.linalg.cholesky(GMM.base[k].Sig_map))
                    for k in range(self.K)]
        self.lam = [Linv_sig[k].T @ Linv_sig[k] for k in range(self.K)]
        self.Llam = [np.linalg.cholesky(self.lam[k]) for k in range(self.K)]
        self.log_det_lam = [np.sum(np.log(np.diag(self.Llam[k])))
                            for k in range(self.K)]  # in fact 1/2 log-det

        # posterior params
        self.alpha = np.ones(self.K) * prior.alpha
