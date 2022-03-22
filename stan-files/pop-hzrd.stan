// random slope and intercept

data {
    int N; // no# observations
    int H; // no# nonlinear bases
    int K; // no# groups
    
    // inputs
    matrix[N, 2] Phi; // linear bases
    matrix[N, H] Psi; // nonlinear bases
    int c[N]; // group labels
    // outputs
    vector[N] y;

    // test inputs
    int Nt; // no# test observations
    matrix[Nt, 2] Phi_test; // linear bases
    matrix[Nt, H] Psi_test; // nonlinear bases
    int c_test[Nt]; // group labels 
    // test outputs
    vector[Nt] y_test;

    // visualise
    int Nxx; // no# visualise observations
    matrix[Nxx, 2] Phi_xx; // linear bases
    matrix[Nxx, H] Psi_xx; // nonlinear bases
}


parameters {
    // intercept and slope
    vector[K] intercepts;
    vector[K] slopes;

    // nonlinear coefficients (tied)
    vector[H] beta;

    // noise
    real<lower=0> sigma;
    
    // parent nodes (hyperpriors)
    vector[2] mu_alpha;
    vector<lower=0>[2] sigma_alpha;
}


model {
    vector[N] yhat;
    
    // hyperpriors
    mu_alpha ~ normal([0, 1.5], [2, .5]);
    sigma_alpha ~ inv_gamma(1, 1);
    
    // priors
    intercepts ~ normal(mu_alpha[1], sigma_alpha[1]);
    slopes ~ normal(mu_alpha[2], sigma_alpha[2]);
    beta ~ cauchy(0, .1); // sparse
    sigma ~ inv_gamma(3, .8);
    
    // likelihood
    for(i in 1:N) {
        yhat[i] = Phi[i, 1] * intercepts[c[i]] + Phi[i, 2] * slopes[c[i]]
            + dot_product(Psi[i], beta);
    }

    
    y ~ normal(yhat, sigma);
}

generated quantities {
    // TODO: fix this
    
    // sample to visualise
    // matrix[Nxx, K] yhat_xx;
    // for(k in 1:K) {
    //     for(i in 1:Nxx) {
    //         yhat_xx[k, i] = Phi_xx[i, 1]*intercepts[k] 
    //             + Phi_xx[i, 2]*slopes[k] + dot_product(Psi_xx[i], beta);
    //     }
    // }
    // posterior predictive likelihood
    // vector[Nt] yhat_test;
    // vector[Nt] lpy_test;
    // for(i in 1:Nt) {
    //     yhat_test[i] = Phi_test[i, 1]*intercepts[c_test[i]] 
    //         + Phi_test[i, 2]*slopes[c_test[i]]
    //         + dot_product(Psi_test[i], beta);

    //     lpy_test[i] = normal_lpdf(y_test[i] | yhat_test[i], sigma);   
    // }
}
