function y = logWishart(Sigma, W, v)
% log pdf Wishart
d = length(Sigma);
B = -0.5*v*logdet(W)-0.5*v*d*log(2)-logMvGamma(0.5*v,d);
y = B+0.5*(v-d-1)*logdet(Sigma)-0.5*trace(W\Sigma);
