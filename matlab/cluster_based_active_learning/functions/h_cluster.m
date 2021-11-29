function [u, ch] = h_cluster(X, varargin)

% Builds a hierarchial clustering using the (stock) MATLAB linkage function
% [agglomerative clustering using ward's linkage and euclidean distance]
% Records clusters as they merge; output clusters start at the root node
% The two child nodes, accosiated with each new node, are recorded for reference
%
% Code written by Lawrence Bull at the University of Sheffield
%
% syntax:       [u, ch] = h_cluster(x)
%
% inputs:       [X] is a [N x d] matrix, containing N observations of d-dimesional
%               input data
%
%               'max_cluster' (optional name-value pair) determines the 
%               maximum number of clusters defined. the default value (N-1) 
%               considers clusters down to single observations
%
% outputs:      {u} is a cell containing N-1 arrays, each represents a node in the
%               hierarchical tree, containing the data indicies accosiated with each cluster
%
%               [ch] is a [2 x N-1] matrix, containing the 2 child nodes for each
%               cluster, u. 

p = inputParser;
addRequired(p,'X');
addOptional(p,'max_clusters', length(X)-1)
parse(p,X,varargin{:});

% build hierarcical tree
Z = linkage(p.Results.X, 'ward', 'euclidean');

% store each cluster as it forms, using Z info
% initialise cell of nodes (clusters)
N = length(p.Results.X); % total number of inputs
u = cell(1,N-1); % number of times you can merge, from singleton obvs.
ch = zeros(2,N-1); % record the two respective children for each cluster as it forms

% save input index groups as each cluster froms
% start by merging singlton obvs
% use the output of linkage (Z) to build clusters.
for m = 1:length(u)
    % join parts p1 & p2 of the clustering together using Z info:   
    % define indicies in p1
    if Z(m,1) <= N % if they are single indices
        p1 = Z(m,1);
        c1=p1; % record point index (cluster number) for the frist singleton child
    else % if they are already a cluster, draw from the clusters already formed
        p1 = u{Z(m,1)-N};
        c1 = 2*N-Z(m,1); % record the cluster number for the first child cluster
    end
    % define indicies in p2    
    if Z(m,2) <= N % if they are single indices
        p2 = Z(m,2);
        c2=p2; % record point index (cluster number) for the second singleton child
    else % if they are already a cluster, draw from the clusters already formed
        p2 = u{Z(m,2)-N};
        c2 = 2*N - Z(m,2); % record the cluster number for the second child cluster
    end
    % join 2 gorups of data indicies
    u{m} = [p1;p2]; % combine two parts to make the new cluster
    ch(:,m) = [c1;c2]; % list the two children for each new cluster
end
% flip the order, so it starts at root
u = flip(u); % NOTE, this flips the meaning of the index..
ch = flip(ch,2);
% i = 1 is now the first split, not the first merge
% index here is the CLUSTER NUMBER: i = 1 is the root node, etc.

% LIMIT the number of cluster considered (speed considerations), 
% limits the search accross nodes for new prunings
u = u(1:p.Results.max_clusters);
ch = ch(:,1:p.Results.max_clusters);
end