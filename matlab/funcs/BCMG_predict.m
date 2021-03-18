function [y_pred, py_xD, px_yD, py_D, lpx_D] = BCMG_predict(xtest, theta)
% Bayes classification
% Code written by Lawrence Bull

% store params
mu_n = theta.mu_n;
k_n = theta.k_n; 
v_n = theta.v_n;
S_n = theta.S_n;
lamda = theta.lamda;

% data properties
k = size(mu_n,1);
d = size(xtest,2);

% POSTERIOR PREDICTIVE 'LIKELIHOOD': p(x | y, D)
% uses student_t function
px_yD = zeros(size(xtest,1), k);
lpx_yD = zeros(size(xtest,1), k);
for i = 1:k
    [px_yD(:, i), lpx_yD(:, i)] = student_t(xtest, mu_n(i,:),...
        ((k_n(i) + 1)/(k_n(i)*(v_n(i)-d+1)))*S_n(:,:,i),...
        (v_n(i)-d+1)); 
end

% POSTERIOR PREDICTIVE 'PRIOR': p(y | D)
py_D = lamda;

% log sum exp-trick
b_c = lpx_yD + repmat(log(py_D), size(lpx_yD, 1), 1);
% PREDICTIVE CLASSIFIER
lpx_D = log(sum(exp(b_c - max(b_c, [], 2)), 2)) + max(b_c, [], 2);
% POSTERIOR PREDICTIVE OF THE MODEL: p(x* | D)
lpy_xD = b_c - lpx_D;

% MAP label
[~, y_pred] = max(lpy_xD,[],2);
py_xD = exp(lpy_xD);

end