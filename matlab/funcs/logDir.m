function [lpth] = logDir(th, c_k, alpha)
% prop. to log-pobability of the dirichlet distribution
% Code written by Lawrence Bull
% th - 1 x D vec of mixing prop.
% c_k - 1 x D vec of counts of members per class
% alpha - 1 x D vec of prior per class or 1 x 1 id all same

lpth = sum((c_k+alpha-1).*log(th));