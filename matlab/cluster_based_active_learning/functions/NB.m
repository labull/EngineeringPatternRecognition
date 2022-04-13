function [y_pred] = NB(x, y, xtest)

% Naive bayes classification
% training/prediction
% 
% Note: regularisation is naturally provided as the computation of the 
% liklihood and the prior are both posterior predictive (see in code) 
%
% Code written by Lawrence Bull at the University of Sheffield 

% NORMALISE according to the trianing data
loc = mean(x);
sdev = std(x);
x = (x - loc)./(sdev);
xtest = (xtest - loc)./(sdev);

% properties of the training data
d = size(x,2); % dimensionality
n = size(x,1); % size

% split training data into class groups
C = unique(y); % value of each class
k = length(C); % number of classes

xc = cell(1,k); % cell to contain data by class
for i = 1:k
    c = C(i);
    xc{i} = x(y == c, :);
end

%--------------------------------------------------------------------------
% TRAIN
%--------------------------------------------------------------------------
% PRIOR calculation
% posterior predictive for the 'prior' (dirichlet catagorical model)

% hyperparameter
alpha = n/k;

p_cD = []; % prior
for i = 1:k
     p_cD = [p_cD, (alpha + size(xc{i}, 1))/(k*alpha + n)]; %#ok<AGROW> MAP 
end

% LIKLIHOOD calculation
% the posterior predictive liklihood

% set hyperparameters
mu_o = 0;
k_o = 1;
v_o = 1;
sig2_o = 1;

% calculate sample mean
mu = zeros(k, d);
for i = 1:k
    for j = 1:d
        mu(i,j) = mean(xc{i}(:,j));
    end
end

% init.
mu_n = zeros(k, d); 
k_n = zeros(k, d);
v_n = zeros(k, d);
sig2_n = zeros(k, d);
sigma2_n = zeros(k,d);

% find post. pred. params for each class, for each feature
for i = 1:k
    for j = 1:d
        nc = size(xc{i}, 1); 
        % note: use 'nc' this is the number of istances in each class
        k_n(i,j) = k_o + nc;
        mu_n(i,j) = (k_o*mu_o + nc*mu(i,j))/k_n(i,j); 
        % note: post. mean of mu
        v_n(i,j) = v_o + nc;
        sig2_n(i,j) = (1/v_n(i,j))*(v_o*sig2_o + sum((xc{i}(:,j) - mu(i,j)).^2) ...
            + (nc*k_o/(k_o + nc))*(mu_o - mu(i,j))^2);  
        sigma2_n(i,j) = v_n(i,j)*sig2_n(i,j)/(v_n(i,j) - 2);  
        % note: post. mean of variance
    end
end

%--------------------------------------------------------------------------
% PREDICT
%--------------------------------------------------------------------------
% evaluate the (log) liklihood for each of test data
% p_xyD = zeros(length(xtest), k);
lp_xyD = zeros(length(xtest), k); % (log-liklihood)
for i = 1:k
    p_kd = [];
    for j = 1:d
        p_kd = [p_kd, ...
            pp_lklhd(xtest(:,j), mu_n(i,j), k_n(i,j), ...
            v_n(i,j), sig2_n(i,j))]; %#ok<AGROW>
    end
%     p_xyD(:, i) = prod(p_kd, 2);
    lp_xyD(:, i) = sum(log(p_kd),2);
end

% POSTERIOR
% p_yxD = zeros(size(xtest,1), k); % posterior
lp_yxD = zeros(size(xtest,1), k); % log-space
for i = 1:size(xtest,1)
    lp_yxD(i,:) = lp_xyD(i,:) + log(p_cD);
%     p_yxD(i,:) = p_xyD(i,:).*p_cD;
end
[~, y_pred] = max(lp_yxD,[],2); % argmax from the log-space

% % NORMALISE the posterior, so that the value's sum to 1
% for i = 1:length(xtest)
%     p_yxD(i,:) = p_yxD(i,:)/sum(p_yxD(i,:));
% end
end

function p_xyD = pp_lklhd(x, mu_n, k_n, v_n, sig2_n)
% function for the posterior predictive (pp) liklihood

p_xyD = (v_n/2)^1/2*...%(gamma((v_n + 1)/2)/gamma(v_n/2))*... [ratio of gamma distributions is approximated]
    (k_n/((k_n+1)*pi*v_n*sig2_n))^1/2 *...
    (1 + (k_n*(x - mu_n).^2)/((k_n + 1)*v_n*sig2_n)).^(-(v_n + 1)/2);
end