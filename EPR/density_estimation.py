import numpy as np
from scipy.special import gammaln, digamma
from scipy.stats import invwishart, multivariate_normal, dirichlet

class student_t:
    # multi-variate student-t 
    def __init__(self, loc, scale, df):
        self.mu = loc
        self.v = df
        self.S = scale
        self.cov = self.v / (self.v - 2) * self.S
        self.D = self.S.shape[0]
        self.L = np.linalg.cholesky(self.S)
        
    def logpdf(self, X):       
        # KPM pg. 47
        res = X - self.mu
        Lu = np.linalg.solve(self.L, res.T)
      
        lp = (gammaln(self.v/2 + self.D/2) - gammaln(self.v/2) - 
              self.D/2*np.log(np.pi) - self.D/2*np.log(self.v) - 
              sum(np.log(np.diag(self.L))) - 
              (self.v+self.D)/2 * np.log(1 + 1/self.v * np.sum(Lu.T*Lu.T, 1)))       
        return lp
    
    def entropy(self):
        h = (- gammaln(self.v/2 + self.D/2) + gammaln(self.v/2) + 
             self.D/2 * self.v*np.pi + (self.v/2 - self.D/2) * 
             (digamma(self.v/2 - self.D/2) - digamma(self.v/2)) + 
             np.sum(np.log(np.diag(self.L))))
        return h
    
    def rvs(self, n):
        # sample from mv-t
        n = int(n)
        u = np.random.chisquare(self.v, n)/self.v
        y = multivariate_normal(np.zeros(self.D), self.S).rvs(n)
        return self.mu + y/np.sqrt(u)[:,None]
        

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
        # M-step MAP EM (conditionals)
        # r is a vector of weighted observations (column k of py_X)
        self.N = np.sum(r)
        self.xm = np.sum(r[:,np.newaxis] * X, 0)/self.N
        self.mn = (self.k0/(self.k0 + self.N)*self.m0 
                   + self.N/(self.k0 + self.N)*self.xm)
        self.kn = self.k0 + self.N
        self.vn = self.v0 + self.N
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
        
    def post_logpdf(self, mu, Sig):
        # define log-pdf of posterior joint w scipy
        # KPMurphy pg. 135
        pmu = multivariate_normal(self.mn, Sig/self.kn)
        pSig = invwishart(self.vn, self.Sn)
        lpmuSig = pmu.logpdf(mu) + pSig.logpdf(Sig)
        return lpmuSig
    
    def theta_sample(self, n):
        # hierarchically sample Sig, mu
        Sig = [[]] * n
        mu = np.empty([n, self.D])
        for i in range(n):
            Sig[i] = invwishart(self.vn, self.Sn).rvs(1)
            mu[i, :] = multivariate_normal(self.mn, Sig[i]/self.kn).rvs(1)
        return mu, Sig
        
    def predict(self, Xt):
        # posterior-predictive given data Xt
        m = self.mn
        S = (self.kn + 1)/(self.kn*(self.vn - self.D + 1)) * self.Sn
        nu = self.vn - self.D + 1
        self.post_pred = student_t(m, S, nu) # save post. pred. object
        lp = self.post_pred.logpdf(Xt)
        return lp
    
    def x_sample(self, n):
        # sample observations from predictive
        n = int(n)
        # if post_pred exists already
        if hasattr(self, 'post_pred'):
            samps = self.post_pred.rvs(n)
        else:
            m = self.mn
            S = (self.kn + 1)/(self.kn*(self.vn - self.D + 1)) * self.Sn
            nu = self.vn - self.D + 1
            post_pred = student_t(m, S, nu) # save post. pred. object
            samps = post_pred.rvs(n)     
        return samps 


class mixture:
    # mixture model
    def __init__(self, K, base_dist, prior):
        self.K = K # number of components
        # list of conditionals
        self.base = [base_dist(prior) for _ in range(self.K)]
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
        self.alpha_n = np.empty(self.K) # posterior alpha
        for k in range(self.K): # update for each class
            # cluster
            self.base[k].train(X[Y==self.labels[k],:])
            # dirichlet
            nk = sum(Y==self.labels[k])
            self.pi_map[k] = ((nk + self.alpha[k] - 1) 
                              /(self.N + np.sum(self.alpha) - self.K))
            # post. alpha            
            self.alpha_n[k] = nk + self.alpha[k]
        # store supervised 'responsibility' (diracs)
        self.r = np.zeros([X.shape[0], self.K]) # labelled data 
        for k in range(self.K):
            self.r[Y == self.labels[k], k] = 1
        # likelihood of supervised training data
        self.lpX = np.concatenate([self.base[k].predict(X[Y==self.labels[k],:]) 
                                   for k in range(self.K)]).sum()
    
    def predict(self, Xt):
        # predict label posterior (MAP) for each cluster
        lp = [self.base[k].predict(Xt) for k in range(self.K)]
        self.lpx_y = np.column_stack(lp) # likelihood of classifier
        # posterior predictive of classifier
        # log-sum-exp
        bc = self.lpx_y + np.log(self.pi_map) # px|y*py
        self.lpx = np.array([np.log(np.sum(np.exp(b_c - np.max(b_c))))
                             + np.max(b_c) for b_c in bc]) # px
        # py|x = (px|y * py) / px
        lpy_x = np.array([bc[i,:] - self.lpx[i] for i in range(bc.shape[0])]) 
        # unlog
        return np.exp(lpy_x)
    
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
        self.lml = [] # log-marg-lik
        self.ll = [] # log-joint-lik

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
            
            # log-lik of base params     
            lpth_D = np.array([self.base[k].post_logpdf(self.base[k].mu_map,
                                                        self.base[k].Sig_map)
                               for k in range(self.K)]).sum()
            # mixing props
            self.alpha_n = np.sum(r, 0) + self.alpha # posterior alpha
            lpi_D = dirichlet.logpdf(self.pi_map, self.alpha_n)

            # append lml/ll
            self.lml = np.append(self.lml, np.sum(self.lpx))
            self.ll = np.append(self.lml, np.sum(self.lpx) + lpth_D + lpi_D)
            print('log-marginal-likelihood:' + '%.4f' % self.lml[-1])
        
        # store the final responsibility
        self.r = r
        
    def semisupervisedEM(self, Xl, Yl, Xu, tol=1e-3):
        # semi-supervised training by EM
        # stack labelled + unlabelled inputs
        X = np.row_stack([Xl, Xu])
        Yl = np.squeeze(Yl)       
        # init model (post) with labelled datatset
        self.train_supervised(Xl, Yl)
        self.N = X.shape[0] # ovrwrt n labelled from train_supervised      
        # init responsility
        rl = self.r
        ru = self.predict(Xu) # unlabelled data    
        r = np.row_stack([rl, ru])
        
        self.ll = [] # joint-log-likelihood
        
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
            
            # log-lik unlabelled
            ll_ul = np.sum(self.lpx)
            # log-lik of base params
            lpth_D = np.array([self.base[k].post_logpdf(self.base[k].mu_map,
                                                        self.base[k].Sig_map)
                               for k in range(self.K)]).sum()
            # mixing props
            self.alpha_n = np.sum(r, 0) + self.alpha # posterior alpha
            lpi_D = dirichlet.logpdf(self.pi_map, self.alpha_n)
            
            # track log-lik of the joint of the model
            self.ll = np.append(self.ll, ll_ul + self.lpX + lpth_D + lpi_D)
            
            print('log-joint-likelihood:' + '%.4f' % self.ll[-1])       
        
        # store the final responsibility and likelihood 
        self.r # resp. for whole semisupervised set
        # for Xu
        self.r_ul = ru
        self.lpx_ul = self.lpx
        