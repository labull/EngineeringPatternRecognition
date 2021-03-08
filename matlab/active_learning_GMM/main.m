%% GMM active learning demo
clearvars; close all; clc;
addpath('funcs')

% load the demo data
load('demo_data.mat');
% plot
figure(1); colormap('jet');
scatter(D(:,1), D(:,2), D(:,end),D(:,end));
% xlim([-10,25]);ylim([-10,10]);zlim([-10,10])

%% Visualise uncertainty sampling

% split into labelled/unlabelled data
N = size(D,1);
[l_idx] = randperm(N, floor(.5*N));
D_l = D(l_idx, :); % labelled data
D_u = D(setdiff(1:N,l_idx), :); % unlabelled data

x = D_l(:,1:end-1); y = D_l(:,end); % observations and labels
x_u = D_u(:,1:end-1); % unlabelled data

% train / predict GMM
[mu_n, k_n, v_n, S_n, lamda, SigMAP, muMAP] = BCMG_train(x, y);
[y_pred, py_xD, px_yD, py_D, px_D] = BCMG_predict(x_u, mu_n, k_n, v_n, S_n, lamda);

% select uncertain data given the current model
[q_idx] = uncertain_sample(py_xD, px_D, x_u, size(x_u,1), 50);
% plot
f2 = figure(2);colormap('jet');
scatter(x(:,1), x(:,2), 50, y, '.'); % labelled data
hold on
scatter(x_u(:,1), x_u(:,2),50,'k.') % unlabelled data
plot_clusters(muMAP, SigMAP)
% uncertain queries
scatter(x_u(q_idx,1), x_u(q_idx,2), 20, 'ro', 'LineWidth', 2); hold off;

% print(f2, 'images/toy_data_queries.png', '-dpng', '-r300');

%% Tests with streaming data

clearvars -except D;

rep = 50; % test repeats
f1rs = []; % f1 metric for passive learning (random sampling)
f1al = []; % f1 metric for active learning (uncertainty sampling)

B = 10; % batch size

% *** qn => number of queries per batch 
% (used to define the query budget: total queries = qn * N/B)
qn = 2;

for i = 1:rep    
    % indices for ordered training - test split
    test_frac = 1/2;
    [av_idx, test_idx] = ordered_test_train(D, test_frac);
    D_av = D(av_idx, :); % available data to be sampled from
    D_test = D(test_idx, :);   
    % tests
    [rs, al] = tests(D_av, D_test, B, qn);
    f1rs = [f1rs; rs];
    f1al = [f1al; al];
    disp(i)
end

%% Plot
f3 = figure(3);
e1 = errorbar(1:size(f1al,2), mean(f1al), std(f1al), 'Marker', '.', 'LineStyle', '-');
e1.CapSize = 5;
e1.MarkerSize = 5;
hold on
e2 = errorbar(1:size(f1rs,2), mean(f1rs), std(f1rs), 'Marker', 'x', 'LineStyle', '--');
e2.CapSize = 1; e2.CapSize = 1;
e2.MarkerSize = 5;
hold off
ylabel('f1-score'); xlabel('batch number');
xlim([1 size(f1rs,2)])
legend([e1, e2], {'active learning', 'random sample'},'box', 'off', 'Location', 'SouthEast')
ylim([0.78 1.01]);
xlabel('batch number'); ylabel('f1 score');
title(sprintf('%1.1f %% labelled data',1/(B/qn)*100));
% print(f3, sprintf('images/%1.1f%%_labelled.png',1/(B/qn)*100), '-dpng', '-r300');
