import numpy as np
from scipy.special import gammaln

class student_t:
    # multi-variate student-t 
    def __init__(self, loc, scale, df):
        self.mu = loc
        self.v = df
        self.S = scale
        self.cov = self.v / (self.v - 2) * self.S
        self.D = self.S.shape[0]
        self.V = self.v * self.S
        # chol tricks
        L = np.linalg.cholesky(self.V)
        self.Vinv = np.linalg.inv(L).T @ np.linalg.inv(L)
        self.Lp = np.linalg.cholesky(np.pi * self.V)
        
    def logpdf(self, X):
        lp = [gammaln(self.v/2 + self.D/2) - gammaln(self.v/2) -
              np.sum(np.log(np.diag(self.Lp))) - (self.v/2 - self.D/2) *
              np.log(1 + (x-self.mu) @ self.Vinv @ (x-self.mu)) for x in X]
        return lp
    

class NIW:
    # normal-inverse-wishart
    def __init__(self, prior):
        # init prior
        self.m0 = prior.m0
        self.k0 = prior.k0
        self.S0 = prior.S0
        self.v0 = prior.v0
        #
        self.D = prior.S0.shape[0] # dimensions
        self.N = 0 # number of observations
        
    def train(self, X):
        # update given data X
        self.N = X.shape[0]
        self.xm = X.mean(0)
        self.mn = (self.k0/(self.k0 + self.N)*self.m0 
                   + self.N/(self.k0 + self.N)*self.xm)
        self.kn = self.k0 + self.N
        self.vn = self.v0 + self.N
        S = np.sum([np.outer(x, x) for x in X], axis=0)
        
        self.Sn = (self.S0 + S + self.k0 * np.outer(self.m0, self.m0) 
                   - self.kn * np.outer(self.mn, self.mn))
        # posterior MAP (of the joint)
        self.mu_map = self.mn
        self.Sig_map = self.Sn / (self.vn + self.D + 2)
        
    def M_step(self, X, r):
        # M-step form MAP EM (conditionals)
        # r is a vector of weighted observations (column k of py_x)
        self.n = np.sum(r)
        self.xm = np.sum(r[:,np.newaxis] * X, 0)/self.n
        self.mn = (self.k0/(self.k0 + self.n)*self.m0 
                   + self.n/(self.k0 + self.n)*self.xm)
        self.kn = self.k0 + self.n
        self.vn = self.v0 + self.n
        S = np.sum([r[i]*np.outer(X[i,:], X[i,:]) for i in range(len(r))], 
                   axis=0)
        self.Sn = (self.S0 + S + self.k0 * np.outer(self.m0, self.m0) 
                   - self.kn * np.outer(self.mn, self.mn))
        # posterior MAP (of the joint)
        self.mu_map = self.mn
        self.Sig_map = self.Sn / (self.vn + self.D + 2)
        
    def post_init(self, m):
        # init poster with (offset) priors (to init EM)
        # center offset
        self.mn = m 
        # init post. params from prior
        self.kn = self.k0
        self.Sn = self.S0
        self.vn = self.v0
        
    def predict(self, Xt):
        # posterior-predictive given data Xt
        m = self.mn
        S = (self.kn + 1)/(self.kn*(self.vn - self.D + 1)) * self.Sn
        nu = self.vn - self.D + 1
        self.post_pred = student_t(m, S, nu) # save post. pred. object
        lp = self.post_pred.logpdf(Xt)
        return lp



class mixture:
    # mixture model
    def __init__(self, K, base_dist, prior):
        self.K = K # number of components
        self.base = [base_dist(prior) for _ in range(self.K)] # list of conditionals
        if prior.alpha is None: # set alphas for the mixture
            raise Exception('define alpha(s)')
        elif np.size(prior.alpha) == 1:
            self.alpha = np.tile(prior.alpha, self.K)
        elif np.size(prior.alpha) != K:
            raise Exception('no# alphas mismatch to K')
        else:
            self.alpha = prior.alpha
            
    
    def train_supervised(self, X, Y):
        # update given supervised training data
        self.N = X.shape[0]
        Y = np.squeeze(Y)
        self.labels = np.unique(Y) # labels
        self.pi_map = np.empty(self.K) # map of mixing proportions
        for k in range(self.K): # update for each class
            # cluster
            self.base[k].train(X[Y==self.labels[k],:])
            # dirichlet
            nk = sum(Y==self.labels[k])
            self.pi_map[k] = ((nk + self.alpha[k] - 1) 
                              /(self.N + np.sum(self.alpha) - self.K))
        
        # likelihood of supervised training data
        self.lpX = np.concatenate([self.base[k].predict(X[Y==self.labels[k],:]) 
                                   for k in range(self.K)]).sum()
        
    
    def predict(self, Xt):
        # predict posterior for each cluster
        lp = [self.base[k].predict(Xt) for k in range(self.K)]
        self.lpx_y = np.column_stack(lp) # likelihood of classifier
        # posterior predictive of classifier
        # log-sum-exp
        bc = self.lpx_y + np.log(self.pi_map) # px|y*py
        self.lpx = np.array([np.log(np.sum(np.exp(b_c - np.max(b_c))))
                             + np.max(b_c) for b_c in bc]) # px
        # py|x = (px|y * py) / px
        lpy_x = np.array([bc[i,:] - self.lpx[i] for i in range(bc.shape[0])]) 

        return np.exp(lpy_x) # unlog
        
    
    def EM(self, X, tol=1e-3):
        # train unsupervised via MAP EM
        self.N = X.shape[0]
        # init mixing props
        self.pi_map = self.alpha/np.sum(self.alpha)
        # init posterior each cluster (offsetting the prior)..
        m = X[np.random.choice(self.N, self.K, replace=False), :] # data as mu
        [self.base[k].post_init(m[k,:]) for k in range(self.K)]
        # init responsility
        r = self.predict(X)
        self.lml = []

        # EM iterations
        while (np.size(self.lml) < 5 or 
               abs(sum(self.lml[-1] - self.lml[-5:-1])) > tol):
            # M-step
            for k in range(self.K): # update for each class
                # clusters
                self.base[k].M_step(X, r[:, k]) # update posterior params.
                # dirichlet
                nk = np.sum(r[:,k])
                self.pi_map[k] = ((nk + self.alpha[k] - 1) 
                                  /(self.N + np.sum(self.alpha) - self.K))
                
            # E-step
            r = self.predict(X) # update responsibility
            self.lml = np.append(self.lml, np.sum(self.lpx)) # append lml
            print('log-marginal-likelihood:' + '%.4f' % self.lml[-1])
        
    def semisupervisedEM(self, Xl, Yl, Xu, tol=1e-3):
        # semi-supervised training by EM
        # stack labelled + unlabelled inputs
        X = np.row_stack([Xl, Xu])
        Yl = np.squeeze(Yl)
        self.N = X.shape[0]
        # init model (post) with labelled datatset
        self.train_supervised(Xl, Yl)
        # init responsility
        rl = np.zeros([Xl.shape[0], self.K]) # labelled data 
        for k in range(self.K):
            rl[Yl == self.labels[k], k] = 1
        ru = self.predict(Xu) # unlabelled data    
        r = np.row_stack([rl, ru])
        # log-marginal-likelihood
        self.ll = []
        
         # EM iterations
        while (np.size(self.ll) < 5 or 
               abs(sum(self.ll[-1] - self.ll[-5:-1])) > tol):
            # M-step
            for k in range(self.K): # update for each class
                # clusters
                self.base[k].M_step(X, r[:, k]) # update posterior params.
                # dirichlet
                nk = np.sum(r[:,k])
                self.pi_map[k] = ((nk + self.alpha[k] - 1) 
                                  /(self.N + np.sum(self.alpha) - self.K))
                
            # E-step
            ru = self.predict(Xu) # update resp. for unlabelled
            r = np.row_stack([rl, ru])
            
            # log-likeli of the joint of the model
            ll_ul = np.sum(self.lpx) 
            self.ll = np.append(self.ll, ll_ul + self.lpX)
            print('log-marginal-likelihood:' + '%.4f' % self.ll[-1])
        
        # store the final responsibility and likelihood for Xl
        self.r_ul = ru
        self.lpx_ul = self.lpx