function [f1, f1_k] = fscore_macro(yp, yt)
% the macro f1 score for a classifier
Y = unique(yt); % the label space
% for each class
f1_k = zeros(1,length(Y));
for i = 1:length(Y)
    k = Y(i);
    % underlying positive labels
    ypk = yp(yt == k);
    % true pos. and false negs
    TP = sum(ypk == k);
    FN = sum(ypk ~= k);
    % underlying negative labels
    ypnk = yp(yt ~= k);
    % true negatives false pos.
%     TN = sum(ypnk ~= k);
    FP = sum(ypnk == k);
    % precision and recall
    pre = TP/(TP + FP);
    rec = TP/(TP + FN);
    
    if pre == 0 || rec == 0
        f1_k(i) = 0;
    else 
        f1_k(i) = 2*(pre*rec)/(pre+rec);
    end       
end
f1 = mean(f1_k);
end
    
