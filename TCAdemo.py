import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wishart
from scipy.stats import multivariate_normal as normal
from scipy.linalg import eig
from os.path import join
#
from EPR.transformations import TCA

# product kernel
def prodK(X1, X2):
    K =  np.dot(X1, X2.T)
    return K

#%% make source and target data
# run multiple times to resample different data, and test TCA

K = 2 # number of clusters
D = 10 # dimensionality of the dataset

ms = [np.random.normal((k+1)*50, 30, D) for k in range(K)] # random mus 
mt = [np.random.normal((k+1)*50, 50, D) for k in range(K)] # random mus

# source data
Xs = []
Ys = []
# unlabelled target data (hidden from TCA, just for plotting)
Xt = []
Yt = []
# number of source/target data
ns = 100
nt = 100

# plot in the same space
plt.figure(1, figsize=[4,6], dpi=300)
clr = ['b','m'] # markers
# sample data
for k in range(K):
    C = wishart.rvs(D*10, np.eye(D), 1) # shared cov  
    xs = normal.rvs(ms[k], C, size=ns)
    xt = normal.rvs(mt[k], C, size=nt)
    y = np.tile(k+1, xs.shape[0])
    
    # source
    Xs.append(xs)
    Ys.append(y)
    # target
    Xt.append(xt)
    Yt.append(y)

# labelled source
Xs = np.row_stack(Xs)
Xs = Xs/Xs.std(0) # scale
Ys = np.concatenate(Ys)
# unlabelled target
Xt = np.row_stack(Xt)
Xt = Xt/Xt.std(0) # scale
Yt = np.concatenate(Yt)

# PCA
Zd = 2 # dimensionality of the subspace
# weights
_, w = eig(np.cov(np.row_stack([Xs[0:-1:2], Xt[0:-1:2]]).T)) # train with even
# project
pcaZs = Xs @ w[:,:Zd]
pcaZt = Xt @ w[:,:Zd]

# -- plot
# train (even idxs)
plt.scatter(pcaZs[0:-1:2,0], pcaZs[0:-1:2,1], 
            s=4, c=np.take(clr, Ys[0:-1:2]-1), label='$D_s$')
plt.scatter(pcaZt[0:-1:2,0], pcaZt[0:-1:2,1], s=8, 
            ec=np.take(clr, Yt[0:-1:2]-1), 
            fc= 'none', marker='o', label='$D_t$')
# test (odd idxs)
plt.scatter(pcaZs[1:-1:2,0], pcaZs[1:-1:2,1], 
            s=3, c=np.take(clr, Ys[1:-1:2]-1), label='$D_s$', alpha=.6)
plt.scatter(pcaZt[1:-1:2,0], pcaZt[1:-1:2,1], s=6, 
            ec=np.take(clr, Yt[1:-1:2]-1), 
            fc= 'none', marker='o', label='$D_t$', alpha=.6)

plt.title('principal component analysis (PCA)')
plt.legend()
# save
pth = join('figures', 'TCAdemo_pca.png')
# plt.savefig(pth)

#%% apply TCA

tca = TCA(Xs[0:-1:2], Xt[0:-1:2], prodK) # train with even
tca.solve(Zd) 

tca.test(Xs[1:-1:2], Xt[1:-1:2]) # train with odd

# -- plot transformed data
plt.figure(2, figsize=[4,6], dpi=300)
# train
plt.scatter(tca.Zs[:,0], tca.Zs[:,1], s=4, 
            c=np.take(clr, Ys[0:-1:2]-1), label='$D_s$')
plt.scatter(tca.Zt[:,0], tca.Zt[:,1], s=8, ec=np.take(clr, Yt[0:-1:2]-1), 
            fc= 'none', marker='o', label='$D_t$')
# test
plt.scatter(tca.Zs_test[:,0], tca.Zs_test[:,1], s=3, 
            c=np.take(clr, Ys[1:-1:2]-1), label='$D_s$', alpha=.6)
plt.scatter(tca.Zt_test[:,0], tca.Zt_test[:,1], s=6, 
            ec=np.take(clr, Yt[1:-1:2]-1), 
            fc= 'none', marker='o', label='$D_t$', alpha=.6)

plt.title('transfer component analysis (TCA)')
plt.legend()

# save
pth = join('figures', 'TCAdemo_tca.png')
# plt.savefig(pth)