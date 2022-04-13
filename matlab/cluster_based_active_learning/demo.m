%% IMPORT TOY DATA
clc; clearvars; close all;
addpath('functions')
pth = fullfile('..','funcs','data.mat');
load(pth);
% training
X = x;
Y = y;

figure(1)
scatter(x(:,1),x(:,2),100,y,'.');
colormap('jet'); colorbar;

%% DEMO 1
% single test

% ACTIVE LEARNING

% --------- CLUSTER the pool of available unlabelled data
[u, ch] = h_cluster(X);

% --------- ACTIVE LEARNING -- build the training set -- the DH learner
n = 100; % label budget
B = 3; % batch size
t = n/3; % number of runs
% the DH learner
[xl, z] = DH_AL(u, ch, B, t, Y);

% define the training-set with the DH results
train_idx = xl(:, 1);
x_train  = X(train_idx, :);
y_train = xl(:, 2);

% --------- CLASSIFICATION -- train/predict with niave bayes classification
y_pred = NB(x_train, y_train, x_test);
% calculate classification accuracy
acc = sum(y_pred == y_test)/length(y_test);
fprintf('\n ACTIVE LEARNING classification accuracy: %.2f %%', acc);

% --------- PLOT the training set built by the active learner
% queried data
z_idx = z(:,1); % output of the DH learner is indexed
Z = X(z_idx,:);
% plot
figure(2)
% ground truth labels
scatter(x_train(:,1),x_train(:,2),100,y,'.','DisplayName','target labels');
hold on;
% DH propagated labels
scatter(x_train(:,1),x_train(:,2),60,y_train,'o', ...
    'DisplayName','propagated labels');
% active queries
scatter(Z(:,1),Z(:,2),40,'kx','DisplayName','active queries');
colormap('jet'); colorbar;
hold off
legend()

% PASSIVE LEARNING comparison -- random sample training
% define the training-set by a random sample
train_idx = randperm(size(X,1), n);
x_train = X(train_idx, :);
y_train = Y(train_idx);
% train/predict with niave bayes classification
y_pred = NB(x_train, y_train, x_test);
% calculate classification accuracy
acc = sum(y_pred == y_test)/length(y_test);
fprintf('\n PASSIVE LEARNING classification accuracy: %.2f %%', acc);

%% DEMO 2a:
% RANDOM SAMPLE TRAINING: passive/supervised learning for an increasing 
% label budget 

% parameters
T = 30; % maximum number of runs (max label budget T*B = 600)
B = 20; % batch size
reps = 20; % number of repeats for each experiment

% error
e_rs = [];
for t = 1:T
    % verbose
    fprintf('\nQUERY BUDGET ------ %d', t*B)
    % accuracy for each repeat
    acc = [];
    for r = 1:reps
        % define the training-set by a random sample from availaible data
        train_idx = randperm(size(X,1), t*B);
        x_train = X(train_idx, :);
        y_train = Y(train_idx);
        
        % CLASSIFICATION
        y_pred = NB(x_train, y_train, x_test);
        % record accuracy of prediction
        acc = [acc, sum(y_pred == y_test)/length(y_test)]; %#ok<AGROW>
    end
    e_rs = [e_rs, 1-mean(acc)]; %#ok<AGROW>
end


%% DEMO 2b:
% CLUSTER BASED ACTIVE LEARNING: the DH learner for an increasing label budget

e_al = []; 
for t = 1:T
    % verbose
    acc= [];
    fprintf('\nQUERY BUDGET ------ %d', t*B)
    for r = 1:reps       
        % ------------- INITIAL CLUSTERING -------------- % 
        [u, ch] = h_cluster(X, 'max_clusters', 100);
        
        % -- HIERARCHICAL SAMPLING FOR ACTIVE LEARNING -- %
        % -- the DH active learning algorithm
        [xl, z, p, l] = DH_AL(u, ch, B, t, Y);
        % define training-set (inputs/outputs).
        % Note, first column of xl is input indices, second is class labels
        train_idx = xl(:, 1);
        x_train  = X(train_idx, :);
        y_train = xl(:, 2);
        
        % --------------- CLASSIFICATION ---------------- %       
        y_pred = NB(x_train, y_train, x_test);
        % record accuracy of prediction
        acc = [acc, sum(y_pred == y_test)/length(y_test)]; %#ok<AGROW>
    end
    e_al = [e_al, 1 - mean(acc)];    %#ok<AGROW>
end

% PLOT
figure(3) 
plot(B:B:B*T, e_rs, '--');
hold on
plot(B:B:B*T, e_al, '--');
xlabel('label budget (n)')
ylabel('classification error (e)')
legend('passive learning','active learning')
hold off 
