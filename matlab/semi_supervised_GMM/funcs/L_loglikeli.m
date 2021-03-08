function [log_lik, LL_xl, LL_xu, LL_theta] = L_loglikeli(x_u, x, y, mu_n, k_n, v_n, S_n, lamda, SigMAP, muMAP, alpha, R_k)
% liklihood (for the stopping) criterion over labelled and 
% unlabelled data
% Code written by Lawrence Bull

% data properties
d = size(x,2);
% n = size(x,1);
Y = unique(y);
k = length(Y);
% ---- labelled data
ll_xl = [];
for i = 1:k
    cidx = y==Y(i);
    [~, lpx_y] = student_t(x(cidx,:), mu_n(i,:),...
        ((k_n(i) + 1)/(k_n(i)*(v_n(i)-d+1)))*S_n(:,:,i),...
        (v_n(i)-d+1));
    lpy = log(lamda(i));
    ll_xl = [ll_xl, sum(lpx_y + lpy)]; %#ok<AGROW>    
end
LL_xl = sum(ll_xl);
% ---- unlabelled data
pxu_y =  zeros(size(x_u,1),k);
for i = 1:k
    pxu_y(:, i) = student_t(x_u, mu_n(i,:),...
            ((k_n(i) + 1)/(k_n(i)*(v_n(i)-d+1)))*S_n(:,:,i),...
            (v_n(i)-d+1));
end   
LL_xu = sum(log(sum(pxu_y.*lamda,2)));
% ---- parameters
ll_theta_X = [];
for j = 1:k
    [~, lpm_yD] = student_t(muMAP(j,:), mu_n(j,:),...
    ( 1 / (k_n(j)*(v_n(j)-d+1)) )*S_n(:,:,j),...
    (v_n(j)-d+1) );
    
    % invert matrices
    LSig = chol(SigMAP(:,:,j), 'lower');
    LSn = chol(S_n(:,:,j), 'lower');
    inv_SigMAP = inv(LSig)'*inv(LSig);
    inv_Sn = inv(LSn)'*inv(LSn);
    
    lpS_D = logWishart( inv_SigMAP, inv_Sn,... 
        v_n(j,:) );
    
    ll_theta_X = [ll_theta_X, lpm_yD + lpS_D]; %#ok<AGROW>
end
lplam_D = logDir(lamda, R_k, alpha);
LL_theta = sum(ll_theta_X) + lplam_D;

% sum up terms
log_lik = LL_xl + LL_xu + LL_theta;


end