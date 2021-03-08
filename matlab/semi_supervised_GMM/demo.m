% 8DOF data visual example
% Code written by Lawrence Bull
clearvars
close all; clc;
addpath('funcs')

%% load training and test data
load('data.mat') % the simulated data from the MSSP paper
% x, y -> training data
% x_test, y_test -> test data


Y = unique(y); % label space
% split into labelled unlabelled sets
N = size(x,1);
n = floor(.03*N); % no# labelled data (3 % of training data)
m = size(x,1) - n; % no# of unlabelled data
idx = sort(randperm(N,n));
x_l = x(idx,:); % labelled observations
y_l = y(idx,:);
x_u = x(setdiff(1:N, idx),:); % unlabelled observations (hidden labels)
% y_u = y(setdiff(1:N, idx));
k = length(unique(Y));

figure(1);
CLR = hsv(k);
% plot the labelled/unlabelled/test subsets
% labelled data - colour markers
% unlabelled data - black markers
% test data - (small) red markers
s0 = scatter(x_test(:,1), x_test(:,2), 10, 'r.');
hold on;
s1 = scatter(x_u(:,1),x_u(:,2),9,'k.');
% hold on;
s2 = gscatter(x_l(:,1),x_l(:,2),y_l, CLR,'.',10);
legend([s2;s1;s0], ...
    {'1','2','3','4','5','6', '{x}_{u,i}','{x}^*_i'},...
    'location', 'northoutside', 'orientation', 'horizontal', 'box','off');
xlim([-25 15]);ylim([-12 12]);
xlabel('x_i^{(1)}');ylabel('x_i^{(2)}')
hold off;

%% learn the classifier given the labelled data only

[mu_n, k_n, v_n, S_n, lamda, SigMAP, muMAP] = BCMG_train(x_l, y_l);
% plot the model given the supervised data *only*
figure(2);
s1 = gscatter(x_l(:,1),x_l(:,2),y_l,CLR,'.',10); hold on; % labelled data
% plot the ellipses
for i = 1:k
    plot_clusters(muMAP(i,:), SigMAP(:,:,i))
end
plot_clusters2([0,0], eye(2)) % plot the prior
hold off;
legend(s1, {'1','2','3','4','5','6'},...
    'location', 'south', 'orientation', 'horizontal','box','off');
xlim([-25 15]);ylim([-12 12]);
xlabel('x_i^{(1)}');ylabel('x_i^{(2)}')
title('supervised GMM')
hold off;

% predict
[y_pred] = BCMG_predict(x_test, mu_n, k_n, v_n, S_n, lamda);
% accuracy
acc1 = sum(y_pred == y_test)/size(y_test,1);
fprintf('supervised accuracy: %.2f\n', acc1)

%% update GMM using the unlabelled data

[mu_nu, k_nu, v_nu, S_nu, lamda_u, Sig_MAP, mu_MAP, log_lik] = ...
    GMM_EMupdate(x_u, x_l, y_l, mu_n, k_n, v_n, S_n, lamda, SigMAP, muMAP);
%
figure(3) % check the log-likelihood increased during training
scatter(1:length(log_lik), log_lik,'.')
xlabel('iteration'); ylabel('log likelihood')

% plot the model given the semi-supervised dataset
% labelled *and* unlabelled data
figure(4);
su=scatter(x_u(:,1),x_u(:,2),20,'k.'); hold on;  % unlabelled data
sl=gscatter(x_l(:,1),x_l(:,2),y_l,CLR,'.',10); hold on; % labelled data
su.MarkerEdgeAlpha = 0.5;
% plot the ellipses
theta = 0:0.1:2*pi;
for i = 1:k
    plot_clusters(mu_MAP(i,:), Sig_MAP(:,:,i))
end
plot_clusters2([0,0], eye(2)) % plot the prior
hold off;
legend([sl;su], {'1','2','3','4','5','6','{x}_{ui}'},...
    'location', 'south', 'orientation', 'horizontal','box','off');
xlim([-25 15]);ylim([-12 12]);
xlabel('x_i^{(1)}');ylabel('x_i^{(2)}')
title('semi-supervised GMM')
hold off;

% predict
[y_pred] = BCMG_predict(x_test, mu_nu, k_nu, v_nu, S_nu, lamda_u);
% accuracy
acc2 = sum(y_pred == y_test)/size(y_test,1);
fprintf('semi-supervised accuracy: %.2f\n', acc2)

% performance increase through semi-supervised learning
% accuracy
fprintf('accuracy increase: %.2f%%\n', 100*(acc2-acc1))

