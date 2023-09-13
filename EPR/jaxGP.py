import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from tqdm import trange
import matplotlib.pyplot as plt

@jit
def kernelSE(x1, x2, hyps):
    '''
    squared-exp kernel function
    ''' 
    # reshape so at least 2d (for 1d inputs)
    x1 = jnp.atleast_2d(x1).T
    x2 = jnp.atleast_2d(x2).T
    
    # diffs between all pairs
    r2 = jnp.sum((jnp.expand_dims(x1, 1) - jnp.expand_dims(x2, 0))**2,
                 axis=-1)
    
    # check hyps from dict
    if 'l' in hyps and 'alpha' in hyps:
        l = hyps['l']
        alpha = hyps['alpha']
    else:
        raise Exception('hyps dict needs process '
                        'variance [alpha] length and scale [l]')
        
    # compute kerel matrix
    K = alpha * jnp.exp(-.5 * r2 / (l**2))
    
    return K

@jit
def zero_mean(x):
    '''
    zero mean function
    '''
    return jnp.zeros(x.shape[0])


def log_marginal_likelihood(x, y, kernel, mean, hyps, jitter=1e-6):
    '''
    log marginal likelihood
    '''
    # mean & kernel func of training data
    mx = mean(x)
    Kxx = kernel(x, x, hyps)
    
    # without observation noise? just add jitter
    if hyps.get('sigma') is None:
        sigI = jitter * jnp.eye(Kxx.shape[0])
    # else add observation noise kernel
    else:
        sigI = hyps['sigma']**2 * jnp.eye(Kxx.shape[0]) + jitter * jnp.eye(Kxx.shape[0])
    
    L = jnp.linalg.cholesky(Kxx + sigI)
    a = jnp.linalg.solve(L.T, jnp.linalg.solve(L, (y - mx)))
    
    lml = (-.5*jnp.inner(y - mx, a) 
           - jnp.sum(jnp.log(jnp.diag(L))) 
           - y.shape[0]/2*jnp.log(2*jnp.pi))
    
    return lml


def predict(xp, x, y, kernel, mean, hyps, jitter=1e-6):
    '''
    GP predict
    '''
    # mean & kernel funcs
    mx = mean(x)
    Kxx = kernel(x, x, hyps)
    
    mxp = mean(xp)
    Kpx = kernel(xp, x, hyps)
    Kpp = kernel(xp, xp, hyps)
    
    # without observation noise? just add jitter
    if hyps.get('sigma') is None:
        sigI = jitter * jnp.eye(Kxx.shape[0])
        sigIp = jitter * jnp.ones(Kpp.shape[0])
    # else add observation noise kernel
    else:
        sigI = hyps['sigma']**2 * jnp.eye(Kxx.shape[0])
        sigIp = hyps['sigma']**2 * jnp.ones(Kpp.shape[0])
    
    L = jnp.linalg.cholesky(Kxx + sigI)
    a = jnp.linalg.solve(L.T, jnp.linalg.solve(L, (y - mx)))
    
    f_mu = mxp + jnp.matmul(Kpx, a)
    v = jnp.linalg.solve(L, Kpx.T)
    f_var = Kpp - jnp.matmul(v.T, v)
    y_var = jnp.diag(f_var) + sigIp
    
    return f_mu, jnp.sqrt(f_var), jnp.sqrt(y_var)
    

def MLtypeII(x, y, kernel, mean, init_hyps,
             steps=100, learning_rate=1e-5, plotter=True):
    '''
    maximum likelihood type-II by gradient ascent
    '''
    # convert to list to use arg nums for autodiff
    hyp_keys = init_hyps.keys()
    hyp_vals = jnp.array(list(init_hyps.values()))
    
    @jit
    def objective(hyp_vals, x, y):
        # back into dict
        hyps = dict(zip(hyp_keys, hyp_vals))
        return log_marginal_likelihood(x, y, kernel, mean, hyps)
    
    # @jit
    # def update(hyp_vals, x, y, lr=learning_rate):
    #     # gradient ascent (check for better) **
    #     return hyp_vals + lr * grad(objective)(hyp_vals, x, y, hyp_keys)
    @jit
    def update(hyp_vals, x, y, state, 
               lr=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        '''
        adam optimizer implementation
        
        state: state variables for Adam (initialized with zeros).
        beta1: exponential decay rate for the first moment estimates.
        beta2: exponential decay rate for the second moment estimates.
        epsilon: jitter to prevent division by zero.
        '''
        
        t = state['t'] + 1
        t_coef = jnp.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)
        
        obj_val, grad = value_and_grad(objective)(hyp_vals, x, y)

        m = beta1 * state['m'] + (1.0 - beta1) * grad
        v = beta2 * state['v'] + (1.0 - beta2) * (grad ** 2)

        hyp_vals = hyp_vals + lr * t_coef * m / (jnp.sqrt(v) + epsilon)

        state = {'t': t, 'm': m, 'v': v}

        return hyp_vals, state, obj_val
    
    # init adam states
    state = {'t': 0, 'm': jnp.zeros_like(hyp_vals), 'v': jnp.zeros_like(hyp_vals)}
    # trace objective
    obj_trace = []
    for _ in trange(steps, ncols=60):
        # hyp_vals = update(hyp_vals, x, y)
        hyp_vals, state, obj_val = update(hyp_vals, x, y, state)
        if jnp.isnan(obj_val):
            raise Exception('objective is NaN')
        else:
            obj_trace.append(obj_val)
        
            
    res = dict(zip(hyp_keys, hyp_vals))
    res['trace'] = obj_trace    # also return trace
    
    if plotter == True:
        # plot trace of objective
        plt.plot(obj_trace)
        plt.ylabel('lml'); plt.xlabel('iteration')
        plt.show()
    
    return res

