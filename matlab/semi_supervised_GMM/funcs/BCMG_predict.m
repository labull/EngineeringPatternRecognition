function [y_pred, py_xD, px_yD, py_D, px_D] = BCMG_predict(xtest, mu_n, k_n, v_n, S_n, lamda)
% Bayes classification
% Code written by Lawrence Bull

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

% PREDICTIVE CLASSIFIER
% p_yxD_ = zeros(size(xtest,1), k);
lpy_xD = zeros(size(xtest,1), k); % log posterior
for i = 1:size(xtest,1)
    % log sum exp-trick
    b_c = lpx_yD(i,:) + log(py_D);
    lpy_xD(i, :) = b_c - log(sum(exp(b_c - max(b_c)))) - max(b_c);
end
[~, y_pred] = max(lpy_xD,[],2);
py_xD = exp(lpy_xD);

% POSTERIOR PREDICTIVE OF THE MODEL: p(x* | D)
px_D = NaN(size(xtest,1),1);
for i = 1:size(xtest,1)
    px_D(i, :) = sum(px_yD(i,:).*py_D);
end

end