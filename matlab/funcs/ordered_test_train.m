function [train_idx, test_idx] = ordered_test_train(D, frac)
% split the data into and ordered training and test sets, to assess the
% classification performance for streaming data
% code written by Lawrence bull
n = round(1/frac);
train_idx = [];
test_idx = [];
for i = 1:size(D,1)
    if isreal(i/n) && rem(i/n,1) == 0
        test_idx = [test_idx, i]; %#ok<AGROW>
    else
        train_idx = [train_idx, i]; %#ok<AGROW>
    end
end