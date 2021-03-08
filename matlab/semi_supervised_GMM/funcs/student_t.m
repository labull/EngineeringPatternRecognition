function [px, lpx] = student_t(x, mu, Sig, v)
% student-t pdf
% function follows Murphy's notation (additional parameters)
% [ML: a probabilistic perspective]
% Code by Lawrence Bull 2018
D = size(x, 2);
n = size(x, 1);
V = v*Sig;
% inverse matrix
% Vinv = inv(V);
L = chol(V, 'lower');
Vinv = inv(L)'*inv(L);
px = [];
lpx = [];
for i = 1:n
%     px = [px;...
%         (v/2)^(D/2)*... %(gamma(v/2 + D/2)/gamma(v/2))*...
%         det(pi*V)^(-1/2)*... 
%         (1 + (x(i,:)-mu)*inv(V)*(x(i,:)-mu)')^((-v-D)/2)]; %#ok<AGROW> 
    % compute in log space (prevent underflow)
    lpx = [lpx;...
        (gammaln(v/2 + D/2) - gammaln(v/2)) + ...
        (-1/2)*log(det(pi*V)) + ...
        (-(v+D)/2)*log((1 + (x(i,:)-mu)*Vinv*(x(i,:)-mu)'))]; %#ok<AGROW>
    % out of log space
    px = exp(lpx);
end
end
