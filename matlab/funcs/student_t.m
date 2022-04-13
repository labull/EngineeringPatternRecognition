function [px, lpx] = student_t(x, mu, Sig, v)
% student-t pdf
% function follows Murphy's notation
% [ML: a probabilistic perspective]
% Code by Lawrence Bull 2018
D = size(x, 2);
% n = size(x, 1);

% one-shot
U = chol(Sig);
res = x - mu;
QU = U'\res';


lpx = gammaln((v+D)/2) - gammaln(v/2) - D/2*log(v) - D/2*log(pi) ...
    - sum(log(diag(U))) - (v+D)/2 * log(1 + sum(QU' .* QU', 2)/v);
%     - sum(log(diag(U))) - (v+D)/2 * log(1 + diag(QU' * QU)/v);

px = exp(lpx);

end
