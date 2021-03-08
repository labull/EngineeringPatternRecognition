function [f1rs, f1al] = tests(D_av, D_test, B, qn)
% Applies uncertainty sampling GMM for data arriving in batches of size of
% size B. qn is the number of queries per batch. D_av are the data
% available for training (labelled and unlabelled), while D_test are the
% held-out test data.


N = size(D_av, 1); % number of observations (excluding test set)

% --- RANDOM SAMPLE (PASSIVE) LEARNING
f1rs = []; % f1 metric
D_t = D_av(1:B,:); % training dataset (initialise as the first batch)
D_u = []; % unlabelled data (initialise as empty)
for b = 1:floor(N/B)-2
    disp(b);
    % training inputs/outputs
    x = D_t(:, 1:end-1); 
    y = D_t(:, end);
    % new data batch
    D_u = [D_u; D_av(b*B+1:b*B+B,:)];  %#ok<AGROW> add new batch to unlabelled pool
    x_u = D_u(:, 1:end-1);
    % test set (for performance metric)
    x_test = D_test(1:floor(b*(size(D_test,1) / (N/B))), 1:end-1);
    y_test = D_test(1:floor(b*(size(D_test,1) / (N/B))), end);
    % normalise
    loc = mean([x; D_u(:, 1:end-1)]);
    sdev = std([x; D_u(:, 1:end-1)]);
    x = (x - loc)./(sdev);
    x_u = (x_u - loc)./(sdev); %#ok<NASGU>
    x_test = (x_test - loc)./(sdev);
    % TRAIN parameters classification model
    [mu_n, k_n, v_n, S_n, lamda, ~, ~] = BCMG_train(x, y);
    % SELECT DATA TO SAMPLE
    % random sample 
    q_idx = size(D_u, 1) - randperm(B, qn) + 1; % from the new batch
    % PREDICT test set using the current model
    y_pt = BCMG_predict(x_test, mu_n, k_n, v_n, S_n, lamda);
    % f1score metric
    f1rs = [f1rs, fscore_macro(y_pt, y_test(:,1))]; %#ok<AGROW>  
    % UPDATE TRAINING SET
    D_t = [D_t; D_u(q_idx, :)]; %#ok<AGROW> update training set
    D_u(q_idx, :) = []; % remove queried data from the unlabelled set
end
clearvars -except D_av D_test B qn N f1rs f1rsem f1al f1alem pcn

% --- UNCERTAINTY SAMPLING ACTIVE LEARNING
f1al = []; % f1 metric
D_t = D_av(1:B,:); % training dataset (initialise as the first batch)
D_u = []; % unlabelled data (initialise as empty)
for b = 1:floor(N/B)-2
    disp(b);
    % training inputs/outputs
    x = D_t(:, 1:end-1); 
    y = D_t(:, end);
    % new data batch
    D_u = [D_u; D_av(b*B+1:b*B+B,:)];  %#ok<AGROW> add new batch to unlabelled pool
    x_u = D_u(:, 1:end-1);
    % test set (for performance metric)
    x_test = D_test(1:floor(b*(size(D_test,1) / (N/B))), 1:end-1);
    y_test = D_test(1:floor(b*(size(D_test,1) / (N/B))), end);
    % normalise
    loc = mean([x; D_u(:, 1:end-1)]);
    sdev = std([x; D_u(:, 1:end-1)]);
    x = (x - loc)./(sdev);
    x_u = (x_u - loc)./(sdev);
    x_test = (x_test - loc)./(sdev);
    % TRAIN parameters classification model
    [mu_n, k_n, v_n, S_n, lamda, ~, ~] = BCMG_train(x, y);
    % PREDICT UNLABELLED DATA
    [~, py_xD, ~, ~, px_D] = BCMG_predict(x_u, mu_n, k_n, v_n, S_n, lamda);
    % DEFINE UNCERTAIN POINTS
    [q_idx] = uncertain_sample(py_xD, px_D, x_u, B, qn);
    % PREDICT test set
    y_pt = BCMG_predict(x_test, mu_n, k_n, v_n, S_n, lamda);
    % f1score metric
    f1al = [f1al, fscore_macro(y_pt, y_test(:,1))]; %#ok<AGROW>
    % UPDATE TRAINING SET
    D_t = [D_t; D_u(q_idx, :)]; %#ok<AGROW> update training set
    D_u(q_idx, :) = []; % remove queried data from the unlabelled set
end
clearvars -except D_av D_test B qn N f1rs f1rsem f1al f1alem pcn
end

