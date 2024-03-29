import numpy as np
import time
import matplotlib.pyplot as plt
# file printing
from IPython.display import display, HTML
import inspect
from pygments import highlight
from pygments.lexers import get_lexer_for_filename, PythonLexer
from pygments.formatters import HtmlFormatter

def chinv(A, jit=1e-6):
    # Cholesky inverse
    L = np.linalg.cholesky(A + np.eye(A.shape[0])*jit)
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


def uniBspline(xx, H, compact=False):
    # list of uniform B-splines over range [xx]
    if compact == False:
        # hold-on extrapolation
        # delta = (xx.max() - xx.min()) / (H - 3)
        delta = (xx.max() - xx.min()) / (H - 1)
        xhh = np.arange(xx.min() - (delta * 2), xx.max(), delta)[:-1]
    elif compact == True:
        # extrapolation to zero
        delta = (xx.max() - xx.min()) / (H + 3)
        xhh = np.arange(xx.min(), xx.max(), delta)[:H]
    # functions
    splines = [lambda x, xh=xh: bspline(x, xh, delta) for xh in xhh]
    return splines


class ellipse:
    # points around 2d covariance ellipse
    def __init__(self, mu, cov, points=500, MSD_th=None):
        theta = np.linspace(0, np.pi * 2, points)
        val, vec = np.linalg.eig(cov)
        a = np.arctan(vec[1, 0] / vec[0, 0])    # angle of fist eigenvector
        self.angle_of_rotation = a
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


def plotSTD(ax, x, y_hat, y_std, devs=3, c='k', a=.1, lw=1, label=None):
    ii = np.argsort(x)
    if y_std.size == 1:
        y_std = np.ones_like(y_hat)*y_std
    ax.plot(x[ii], y_hat[ii], c=c, lw=lw, label=label)
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


def fprint(file_path):
    '''
    printing files... (stan files)
    '''
    try:
        with open(file_path, 'r') as file:
            code = file.read()
            lexer = get_lexer_for_filename(file_path) 
            formatter = HtmlFormatter(style='colorful', noclasses=True)
            highlighted_code = highlight(code, lexer, formatter)
            display(HTML(highlighted_code))
    except Exception as e:
        print(f"Error: {e}")
        

def funprint(python_function):
    '''
    printing python functions
    '''
    source_code = inspect.getsource(python_function)
    highlighted_code = highlight(source_code,
                                 PythonLexer(), HtmlFormatter())
    display(HTML(highlighted_code))

        
def Nhistbin(y):
    '''
    freedman-diaconis rule (for hist bin number)
    '''
    if y.size == 0:
        bin_count = 0
    else:
        q1 = np.quantile(y, 0.25)
        q3 = np.quantile(y, 0.75)
        iqr = q3 - q1
        bin_width = (2 * iqr) / (y.shape[0] ** (1 / 3))
        bin_count = int(np.ceil((y.max() - y.min()) / bin_width))
    return bin_count


def plot_dist(ax, x, colour='teal', ground_truth=None):
    '''
    bin-type plotter for distributions
    '''
    ax.hist(x, bins=Nhistbin(x), color=colour,
             alpha=0.6, density=True, rwidth=.8)
    
    if ground_truth is not None:
        ax.plot(ground_truth, 0, 'k^', ms=15, alpha=.8,
                label='ground truth')

    
def auto_corr(ax, trace, lags=60):
    '''
    autocorrelation for mcmc chains [a FCiML]
    '''
    auto_corr = np.ones(lags)
    diff = trace - trace.mean()
    
    for k in range(1, lags):
        auto_corr[k] = (1 / ((trace.shape[0]-k) * trace.var()) 
                        * diff[:-k].T @ diff[k:]).flatten()
    ax.bar(range(lags), auto_corr,
            color='gray', alpha=.6, width=.5)
    ax.set_xlabel('lags'); ax.set_ylabel('auto corr')


def plot_trace(trace, names=None):
    '''
    trace plot, one chain
    '''
    D = trace.shape[1] # no of parameters (cols)
    for d in range(D):
        _, ax = plt.subplots(2)
        # trace plot
        ax[0].plot(trace[:, d], color='teal')
        if names is not None:
            ax[0].set_ylabel(names[d])
        ax[0].set_xlabel('samples')
        # autocorrelation
        auto_corr(ax[1], trace[:, d])
        plt.tight_layout()
        plt.show()

