function [mu_n, k_n, v_n, S_n, lamda, SigMAP, muMAP] = BCMG_train(x, y)
% Learn the distribution for a Bayesian mixture of Gaussian components
% Code written by Lawrence Bull

% data properties
d = size(x,2);
n = size(x,1);
Y = unique(y);
k = length(Y);

xc = cell(1,k); % cell of data by class
for i = 1:k
    xc{i} = x(y == Y(i), :);
end

% PARAMETERS OF POSTERIOR PREDICTIVE
% FOR THE LABELS / MIXING PROPORTIONS: p(y | D)
% hyperparameters
alpha = ones(1, k).*n/k;
lamda = [];
for i = 1:k
     lamda = [lamda,...
         (alpha(i) + size(xc{i}, 1))/(sum(alpha) + n)]; %#ok<AGROW>
end

% PARAMATERS OF POSTERIOR PREDICTIVE 
% FOR THE MIXTURE OF GAUSSIANS: p(x | y, D)
% hyperparameters
% set hyperparameters (prior)
mu_0 = zeros(1, d);
k_0 = 1;
v_0 = d;
S_0 = diag(ones(1, d));
% empirical parameters
mu = NaN(k, d);
S = NaN(d, d, k);
for i = 1:k
    mu(i,:) = mean(xc{i}, 1); % sample mean for each class
    S(:,:,i) = xc{i}'*xc{i}; % scatter matrix (uncentered sum of squares)
end
% params of posterior predictive
mu_n = NaN(k, d); 
k_n = NaN(k, 1);
v_n = NaN(k, 1);
S_n = NaN(d, d, k);
SigMAP = NaN(d,d,k);
muMAP = NaN(k,d);
for i = 1:k
    ny = size(xc{i}, 1);
    mu_n(i,:) = (k_0/(k_0+ny))*mu_0 + (ny/(k_0+ny))*mu(i,:);
    k_n(i,:) = k_0 + ny;
    v_n(i,:) = v_0 + ny;
    S_n(:,:,i) = S_0 + S(:,:,i) + ...
        k_0*(mu_0'*mu_0) - k_n(i,:)*(mu_n(i,:)'*mu_n(i,:));
    % MAP estimates of mu and Sigma
    SigMAP(:,:,i) = S_n(:,:,i)/(v_n(i) + d + 1);
    muMAP(i,:) = mu_n(i,:);
end
end



