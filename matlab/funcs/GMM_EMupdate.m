function [theta, log_lik] = GMM_EMupdate(x_u, x, y, theta)
% Learn the distribution for a mixture of Gaussians
% UPDATE parameters by MAP expectation maximisation (EM): 
% labelled + unlabelled data 
% Code written by Lawrence Bull

% data properties
d = size(x,2);
Y = unique(y);
k = length(Y);
n = size(x,1); % # labelled data
m = size(x_u,1); % # unlabelled data
% stack data together
X = [x;x_u];
N = n + m;
% PRIOR
% parameters from labelled data become the priors (empirical Bayes)
alpha = theta.lamda*N; % multiply by n + m to define prior counts
mu_0 = theta.mu_n;
k_0 = theta.k_n;
v_0 = theta.v_n;
S_0 = theta.S_n;
% init joint liklihood
% log_lik = [inf; inf; inf; realmax]; % init while loop
log_lik = [];
t = 0; % it. counter

% while log-L in inc.
while length(log_lik) <= 3 || sum(log_lik(end)-log_lik(end-3:end)) > 1e-3
    t =t+1;
    % E-step
    % repsonsibilty matrix p(y | x, D)
    % unlabelled
    [~, r] = BCMG_predict(x_u, theta);
    % labelled
    p = zeros(n,k);
    for i = 1:n; p(i,Y==y(i)) = 1; end
    R = [p;r]; % responsibility over the whole dataset (X)
    R_k = sum(R, 1); % counts
    
    % M-step MAP{p(theta | D)}
    % mixing params
    theta.lamda = (R_k + alpha - 1)/(N + sum(alpha) - k);
    % cluster params
    for j = 1:k        
        % parameters from R matrix
        x_k = (sum(r(:,j).*x_u,1) + sum(p(:,j).*x,1))./R_k(j);
        sk = NaN(d,d,N);
        for i = 1:N
            sk(:,:,i) = R(i,j)*(X(i,:)'* X(i,:));
        end
        S_k = sum(sk,3);
        % cluster params
        theta.mu_n(j, :) = (k_0(j)/(k_0(j)+R_k(j))) * mu_0(j,:) + ...
            (R_k(j)/(k_0(j)+R_k(j))) * x_k;
        theta.k_n(j,:) = k_0(j) + R_k(j);
        theta.v_n(j,:) = v_0(j) + R_k(j);
        theta.S_n(:,:,j) = S_0(:,:,j) + S_k + ...
                k_0(j)*(mu_0(j,:)'*mu_0(j,:)) ...
                - theta.k_n(j)*(theta.mu_n(j,:)'*theta.mu_n(j,:));
        % MAPS
        theta.SigMAP(:,:,j) = theta.S_n(:,:,j)/(theta.v_n(j) + d + 2);
        theta.muMAP(j,:) = theta.mu_n(j,:);
        
    end
    % joint liklihood of the parameters (itt. t)
    log_lik_t = L_loglikeli(x_u, x, y, theta, R_k);
    
    log_lik = [log_lik, log_lik_t]; %#ok<AGROW>
    fprintf('iteration %i log-likelihood: %s \n', t, log_lik_t);
end
end
