function y = logdet(A)
% logdet(A) PSD
[U,p] = chol(A);
if p > 0
    y = -inf;
else
    y = 2*sum(log(diag(U)));
end
