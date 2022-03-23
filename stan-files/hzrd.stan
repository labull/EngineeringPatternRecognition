// varying slope/intercept and tied spline model

data {
    int N; // no# observations
    int H; // no# nonlinear bases
    
    // inputs
    matrix[N, 2] Phi; // linear bases
    matrix[N, H] Psi; // nonlinear bases
    // outputs
    vector[N] y;

    // test inputs
    int Nt; // no# test observations
    matrix[Nt, 2] Phi_test; // linear bases
    matrix[Nt, H] Psi_test; // nonlinear bases
    // test outputs
    vector[Nt] y_test;

    // visualise
    int Nxx; // no# visualise observations
    matrix[Nxx, 2] Phi_xx; // linear bases
    matrix[Nxx, H] Psi_xx; // nonlinear bases
}


parameters {
    // intercept and slope
    real intercept;
    real slope;

    // nonlinear coefficients
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
    intercept ~ normal(mu_alpha[1], sigma_alpha[1]);
    slope ~ normal(mu_alpha[2], sigma_alpha[2]);
    beta ~ cauchy(0, .1); // sparse
    sigma ~ inv_gamma(3, 0.8);

    // likelihood
    for(i in 1:N) {
        yhat[i] = Phi[i, 1] * intercept + Phi[i, 2] * slope 
            + dot_product(Psi[i], beta);
    } 

    y ~ normal(yhat, sigma);
}

generated quantities {
    // sample to visualise
    vector[Nxx] yhat_xx;
    for(i in 1:Nxx) {
        yhat_xx[i] = Phi_xx[i, 1]*intercept + Phi_xx[i, 2]*slope 
            + dot_product(Psi_xx[i], beta);   
    }
    
    // posterior predictive likelihood
    vector[Nt] yhat_test;
    for(i in 1:Nt) {
        yhat_test[i] = Phi_test[i, 1]*intercept + Phi_test[i, 2]*slope 
            + dot_product(Psi_test[i], beta);
    }
    real lpy_test = normal_lpdf(y_test | yhat_test, sigma);
}
