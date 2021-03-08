function y = logMvGamma(x,d)
% logmvGamma
s = size(x);
x = reshape(x,1,prod(s));
x = bsxfun(@plus,repmat(x,d,1),(1-(1:d)')/2);
y = d*(d-1)/4*log(pi)+sum(gammaln(x),1);
y = reshape(y,s);
