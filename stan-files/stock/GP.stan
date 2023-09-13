functions {
   vector gp_pred_rng(array[] real xp,
                      vector y,
                      array[] real x,
                      real alpha,
                      real l,
                      real sigma) {
                          
    vector[size(xp)] fp;
    int Np;
    int N;
    Np = size(xp);
    N = rows(y);
        {            
            matrix[N, N] L_Sigma;
            vector[N] Kdiv_y;
            matrix[N, Np] k_xxp;
            matrix[N, Np] v;
            vector[Np] fp_mu;
            matrix[Np, Np] fp_cov;
            matrix[Np, Np] jit;
            matrix[N, N] Sigma;
            
            Sigma = gp_matern32_cov(x, alpha, l);
            Sigma = add_diag(Sigma, rep_vector(square(sigma), N));
            L_Sigma = cholesky_decompose(Sigma);
            
            Kdiv_y = mdivide_left_tri_low(L_Sigma, y);
            Kdiv_y = mdivide_right_tri_low(Kdiv_y',L_Sigma)';
            
            k_xxp = gp_matern32_cov(x, xp, alpha, l);
            fp_mu = (k_xxp' * Kdiv_y);
            v = mdivide_left_tri_low(L_Sigma, k_xxp);
            fp_cov = gp_matern32_cov(xp, alpha, l) - v' * v;
            jit = diag_matrix(rep_vector(1e-9,Np));
            
            fp = multi_normal_rng(fp_mu, fp_cov + jit);
        }
        
        return fp;
    }
}

data {
    int<lower=1> N;
    int<lower=1> Np;
    array[N] real x;
    vector[N] y;
    array[Np] real xp;
    vector[Np] y_test;
    
    // prior hyperparams
    vector<lower=0>[2] l0;
    vector<lower=0>[2] alpha0;
    vector<lower=0>[2] sigma0;
    
}

transformed data {
    vector[N] zeros;
    zeros = rep_vector(0, N);
}

parameters {
    real<lower=0> l;
    real<lower=0> alpha;
    real<lower=0> sigma;    
}

model {
    matrix[N, N] L;
    {
        matrix[N, N] cov;
        cov = gp_matern32_cov(x, alpha, l);
        cov = add_diag(cov, rep_vector(square(sigma), N) + 1e-9);
        L = cholesky_decompose(cov);
    }
    
    l ~ normal(l0[1], l0[2]);
    alpha ~ normal(alpha0[1], alpha0[2]);
    sigma ~ normal(sigma0[1], sigma0[2]);
    
    y ~ multi_normal_cholesky(zeros, L);
}

generated quantities {
    vector[Np] fp;
    vector[Np] yp;
    real lpY;
    
    fp = gp_pred_rng(xp, y, x, alpha, l, sigma);
    for (n in 1:Np) {
        yp[n] = normal_rng(fp[n], sigma);
    }
    lpY = normal_lpdf(y_test | fp, rep_vector(sigma, Np));
}
