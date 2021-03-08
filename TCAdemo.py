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
D = 50 # dimensionality of the dataset

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

et = []

clr = ['b','m'] # markers

# plot in the same space
plt.figure(1, figsize=[4,6], dpi=300)
# sample data
for k in range(K):
    C = wishart.rvs(D*10, np.eye(D), 1) # shared
    
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
Xs = (Xs - Xs.mean(0))/Xs.std(0)
Ys = np.concatenate(Ys)

# unlabelled target
Xt = np.row_stack(Xt)
Xt = (Xt - Xt.mean(0))/Xt.std(0)
Yt = np.concatenate(Yt)


# PCA
Zd = 2 # dimensionality of the subspace
# weights
_, w = eig(np.cov(np.row_stack([Xs, Xt]).T)) 
# project
pcaZs = Xs @ w[:,:Zd]
pcaZt = Xt @ w[:,:Zd]
# plot
plt.scatter(pcaZs[:,0], pcaZs[:,1], s=4, c=np.take(clr, Ys-1), label='$D_s$')
plt.scatter(pcaZt[:,0], pcaZt[:,1], s=8, 
            ec=np.take(clr, Yt-1), fc= 'none', marker='o', label='$D_t$')
plt.title('principal component analysis (PCA)')
plt.legend()

# save
pth = join('figures', 'TCAdemo_pca.png')
# plt.savefig(pth)

#%% apply TCA

tca = TCA(Xs, Xt, prodK)
tca.solve(Zd) 

# plot transformed data
plt.figure(2, figsize=[4,6], dpi=300)

plt.scatter(tca.Zs[:,0], tca.Zs[:,1], s=4, c=np.take(clr, Ys-1), label='$D_s$')
plt.scatter(tca.Zt[:,0], tca.Zt[:,1], s=8, 
            ec=np.take(clr, Yt-1), fc= 'none', marker='o', label='$D_t$')
plt.title('transfer component analysis (TCA)')
plt.legend()

# save
pth = join('figures', 'TCAdemo_tca.png')
# plt.savefig(pth)