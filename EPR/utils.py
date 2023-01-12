import numpy as np
import time

def chinv(A):
    # Cholesky inverse
    L = np.linalg.cholesky(A)
    Linv = np.linalg.inv(L)
    return Linv.T @ Linv


def bspline(x, xh, delt):
    # knots
    xh1 = xh + 1 * delt
    xh2 = xh + 2 * delt
    xh3 = xh + 3 * delt
    xh4 = xh + 4 * delt
    bs = []
    for xx in x:
        if xh <= xx < xh1:
            u = (xx - xh) / delt
            bh = 1 / 6 * u ** 3
        elif xh1 <= xx < xh2:
            u = (xx - xh1) / delt
            bh = 1 / 6 * (1 + 3 * u + 3 * u ** 2 - 3 * u ** 3)
        elif xh2 <= xx < xh3:
            u = (xx - xh2) / delt
            bh = 1 / 6 * (4 - 6 * u ** 2 + 3 * u ** 3)
        elif xh3 <= xx < xh4:
            u = (xx - xh3) / delt
            bh = 1 / 6 * (1 - 3 * u + 3 * u ** 2 - u ** 3)
        else:
            bh = 0
        bs.append(bh)
    return np.array(bs)


def uniBspline(xx, H):
    # list of uniform B-splines over range xx
    delta = (xx.max() - xx.min()) / (H - 1)
    xhh = np.arange(xx.min() - (delta * 2), xx.max(), delta)[:-1]
    # functions
    splines = [lambda x, xh=xh: bspline(x, xh, delta) for xh in xhh]
    return splines


class ellipse:
    # points around 2d covariance ellipse
    def __init__(self, mu, cov, points=500, MSD_th=None):
        theta = np.linspace(0, np.pi * 2, points)
        val, vec = np.linalg.eig(cov)
        a = np.arctan(vec[1, 0] / vec[0, 0])    # angle of fist eigenvector
        # std_dev ellipses
        self.cov_3 = []
        for j in range(3):
            cx = (j+1) * np.sqrt(val[0]) * np.cos(theta)
            cy = (j+1) * np.sqrt(val[1]) * np.sin(theta)
            cr = np.array([[np.cos(a), -np.sin(a)],
                           [np.sin(a), np.cos(a)]]) @ np.vstack((cx, cy))
            self.cov_3.append(cr.T + mu)
        # MSD ellipses - 95%       
        cx = np.sqrt(5.991 * val[0]) * np.cos(theta)
        cy = np.sqrt(5.991 * val[1]) * np.sin(theta)
        cr = np.array([[np.cos(a), -np.sin(a)],
                       [np.sin(a), np.cos(a)]]) @ np.vstack((cx, cy))
        self.MSDellipse_95 = cr.T + mu
        # MSD custom threshold
        if MSD_th is not None:
            cx = np.sqrt(MSD_th * val[0]) * np.cos(theta)
            cy = np.sqrt(MSD_th * val[1]) * np.sin(theta)
            cr = np.array([[np.cos(a), -np.sin(a)],
                           [np.sin(a), np.cos(a)]]) @ np.vstack((cx, cy))
            self.MSD_th = cr.T + mu


class struct(dict):
    """
    Credit to: epool on stackoverflow
    similar to matlab structs
    """
    def __init__(self, *args, **kwargs):
        super(struct, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(struct, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(struct, self).__delitem__(key)
        del self.__dict__[key]


class tic:
    def __init__(self):
        self.t = time.time()
    def toc(self):
        seconds = time.time() - self.t
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print('time: ' + '%d : %d : %d' % (h, m, s))


def plotSTD(ax, x, y_hat, y_std, devs=3, c='k', a=.1, lw=1):
    ii = np.argsort(x)
    ax.plot(x[ii], y_hat[ii], c=c, lw=lw)
    ax.fill_between(x[ii], y_hat[ii] + devs*y_std[ii], y_hat[ii] - devs*y_std[ii],
                    alpha=a, color=c, lw=0, zorder=0)
    
    
def plotF(ax, x, f, ss=50, c='k', a=.1, lw=1, thin=None):
    '''
    plot samples of functions
    '''
    if thin is None:
        # sub sample
        n = f.shape[0]
        y = f[np.random.permutation(n)[:ss]]
    elif thin is not None:
        # thin out f samps
        y = f[1::thin]
    ns = y.shape[0]
    x = np.tile(x, (ns, 1))
    ax.plot(x.T, y.T, c=c, lw=lw, alpha=a, zorder=0)
