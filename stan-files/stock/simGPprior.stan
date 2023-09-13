data {
    // inputs to sim
    int<lower=1> N;
    array[N] real x;    
    // prior hyperparams
    vector<lower=0>[2] l0;
    vector<lower=0>[2] alpha0;
    vector<lower=0>[2] sigma0;
    
}

transformed data {
    vector[N] zeros;
    zeros = rep_vector(0, N);
}

parameters {}

model {}

generated quantities {
    
    real<lower=0> l;
    real<lower=0> alpha;
    real<lower=0> sigma;
    
    l = gamma_rng(l0[1], l0[2]);
    alpha = gamma_rng(alpha0[1], alpha0[2]);
    sigma = gamma_rng(sigma0[1], sigma0[2]);
    
    matrix[N, N] L;
    {
        matrix[N, N] cov;
        cov = gp_matern32_cov(x, alpha, l);
        cov = add_diag(cov, rep_vector(1e-6, N));
        L = cholesky_decompose(cov);
    }
    
    
    vector[N] f;
    vector[N] y;
    f = multi_normal_cholesky_rng(zeros, L);
    for (n in 1:N) {
        y[n] = normal_rng(f[n], sigma);
    }
}
