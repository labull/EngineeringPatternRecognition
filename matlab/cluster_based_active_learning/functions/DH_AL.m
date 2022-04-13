function [XL, z, P, L] = DH_AL(u, ch, B, T, y)

% DH algorithm for cluster based active learning.
% Code impliments the hierarchical active learning algorithm proposed
% by Sanjoy Dasgupta and Daniel Hsu.
% Reference:    'Hierarchical Sampling for Active Learning' 
%               [http://dl.acm.org/citation.cfm?id=1390183]
%
% Code written by Lawrence Bull at the University of Sheffield
%
%
% syntax:       [XL, z, P, L] = DH_AL(u, ch, B, T, y)
%
%
% inputs:       {u} is a cell containing N-1 arrays. each array represents
%               a node in the hierarchical tree, and contains the data indicies 
%               accosiated with each cluster. this variable is an output of
%               the 'h_cluster.m' function.
%
%               [ch] is a [2 x N-1] matrix, containing the 2 children nodes for each
%               cluster, u. this variable is an output of the 'h_cluster.m' function.
%                   
%               B is batch size when running the algorithm. number of queries
%               before potential update of the pruning and labelling.
%               
%               T algoritm run budget; query budget = T x B.
%               
%               [y] N 'hidden' labels for your input data, this must be in the same order
%               as the input observations.
%
%
% outputs:      [XL] is the labelled dataset provided by the DH learner, 
%               including propogated and queried labels. Rows represent 
%               observations. column 1 is the observation index (in X), 
%               column 2 is the class label.
%
%               [z] is a [N x 3] matrix of the queried data. column1 = data index.
%               culumn1 = cluster number it was sampled from, col.3 = label.
%
%               [P] the final pruning of the hierarchical clustering
%
%               [L] the majority label for each cluster in the final
%               pruning


% define N, total number of data
N = length(u{1}); % same as the number of obervations in the root node.
% node weights
Nu = cellfun(@(x) size(x,1), u, 'UniformOutput', false); % data count in each node
wu = cell2mat(Nu)./N;

% store queries
z = []; % queries
uz = cell(1,N-1); % quries according to node
u_ = u; % working version of u (sample pool) to prevent resampling

% initialise node properties
pl = cell(1,length(u)); % proportions of each label, with bounds appended
Aul = cell(1,length(u)); % admissible label sets
Au = zeros(1,length(u)+N); % admissible label logistic array
eu = ones(1,length(u)+N); % errors
Lu = cell(1,length(u)); % empty node majority label cell

P = [1]; %#ok<NBRAK> % set pruning to be the first node only

for t = 1:T
%     fprintf('\nRUN%d: QUERYING DATA POINT:', t); % VERBOSE1
%     fprintf('\nRUN%d: PRUNING: \t %s', t, mat2str(P)); % VERBOSE3
    for b = 1:B
        % ------ SELECT v from P ------ %              
        prop = wu(P); % set proportionality to node weight
        for i = 1:numel(P)
            v = P(i); % cluster index label
            if numel(P)==1 % if its the root node prop = 1 
                coeff = 1;
            elseif isempty(u_{v}) % if no more samples remain set prop = 0
                coeff = 0;              
            elseif ~isempty(u_{v}) % THE SELECT RULE
                coeff = 1 - max(pl{v}(:,end-1)); % coeff. prop. to error in that node
            end
            prop(i) = prop(i)*coeff; % update proportionality to include error coefficient
        end
        
        % select node according to proportionality
        prob = prop/norm(prop,1); % nomalise porportionality to make probability
        cs = cumsum(prob); i = 1:length(prob);
        vi = i(1+sum(cs(end)*rand>cs)); % select a cluster index according tp prob.
                
        % ------ QUERY LABEL ------ % 
        % pick a random query index fom the cluster vi in P
        s = datasample(u_{P(vi)},1);
            
        % remove z from the sample pool, u_, to prevent resempling
        d_i = cellfun(@(x) find(x==s), u_, 'UniformOutput',false);
        for i = 1:numel(u_)
            u_{i}(d_i{i})=[];
        end
        
        l = y(s); % query from hidden labels (y)
%         fprintf(' [%d]',s); % VERBOSE2
        z = [z; s, P(vi), l]; %#ok<AGROW> store the point index, the cluster it was sampled from and it's label

        % ------ UPDATE NODE COUNTS ------ % 
        % for all nodes that contain the new sample (bottom up): porportions & bounds
        % determine wich new node contain index s
        log = cellfun(@(x) ismember(s,x), u, 'UniformOutput',false); % logistic array
        u_i = find(cell2mat(log)==1); % cluster numbers to update, contiaining sample s

        for i = 1:numel(u_i)
            uw = u_i(i); % working cluster number  
            uz{uw} = [uz{uw};s,l]; % organise quried points according to nodes
            nu = size(uz{uw},1); % sample counts in each node

            cl = unique(uz{uw}(:,end)); % classes obsrved in that cluster
            c = sum(uz{uw}(:,end)==cl'); % counts for each of the observed classes
            p_l = (c'./nu); % proportions of each label

            % find the bounds of these estimates
            delta = 1/nu + sqrt((p_l.*(1-p_l))/nu); % confidence interval
            lb = max(p_l-delta,0);
            ub = min(p_l+delta,1);

            pl{uw} = [cl,p_l,lb,ub]; % [label, pl (label count/total count for each node)]
            
            % majority label
            Lu{uw} = pl{uw}(pl{uw}(:,2) == max(pl{uw}(:,2)), 1);           
        end   
    end
    
    % ------ UPDATE ADMISSIBILITIES, ERROR/SCORES ------ %
    % after each batch size, B
    % update admissibilities and error for the nodes containing samples
    u_i = find(cell2mat(cellfun(@isempty, pl, 'UniformOutput', false))==0); % vector containing cluster numbers
    
    for i = 1:numel(u_i)
        uw = u_i(i); % working node number
        % BETA HYPERPARAMETER
        beta = 1.5; 
        p_l = pl{uw}; % extract proportion infromation for the working cluster
        if size(uz{uw},1)>1 % if there is more than one sample check if it's asmissible
            % admissibility
            LHS = p_l(:,end-1);
            RHS = (beta.*p_l(:,end)-1);
            a_l = LHS' > RHS; % logical comparison matrix for inequality results, label L compared to all others
            a_l = a_l.*abs(eye(size(a_l))-1); % delete diagonal so that a label is not considered admissible against itself

            idx = find(sum(a_l) == size(p_l,1) - 1); % asmissible labels succeed the inequality for all other labels...       
            Aul{1,uw} = p_l(idx,1); % store labels in the admiss. cell.
            Au(uw) = ~isempty(idx); % mark admissible nodes in Au cell with a [1]
            % error/score
            eu(uw) = (1 - max(p_l(:,2))); % record error for assigning majority label       
            if isempty(idx) 
                eu(uw) = 1; % where there are no admissible labels, overwrite error = 1
            end
        end       
    end
    % adjust the error for undersampled nodes
    eu(Au==0)=1; % where labels are not admissible, the error==1
    
    % ------ REFINE PRUNING & LABELLING ------ %
    % output pruning labelled same as the input, to re-assign 'P'
    % P becomes the refined set of nodes
    [P, L, XL] = prune_label(P, u, Au, Lu, ch, z, N, eu, wu);
end
end

