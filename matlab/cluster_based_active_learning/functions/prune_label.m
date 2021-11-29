function [P_, L, XL] = prune_label(P, u, Au, Lu, ch, z, N, eu, wu)

% Function to refine the pruning (P) and labelling (L), 
% for the DH active learner: cluster based active learning,
% proposed by Sanjoy Dasgupta and Daniel Hsu.
% Reference:    'Hierarchical Sampling for Active Learning' 
%               [http://dl.acm.org/citation.cfm?id=1390183]
%
% Code written by Lawrence Bull at the University of Sheffield
%
% syntax:       [P_, L, XL] = prune_label(P, u, Au, Lu, ch, z, N, eu, wu)
%
%
% inputs:       [P] input pruning, to be refined (array of node/cluster
%               numbers)
%
%               {u} is a cell containing N-1 arrays. each array represents
%               a node in the hierarchical tree, and contains the data indicies 
%               accosiated with each cluster. this variable is an output of
%               the 'h_cluster.m' function.
%
%               [Au] logistic array, indicating the asmissiblilty of each
%               cluster in the current pruning
%
%               {Lu} cell containing the majority label(s), for each node
%
%               [ch] is a [2 x N-1] matrix, containing the 2 children nodes for each
%               cluster, u. this variable is an output of the 'h_cluster.m' function
%                   
%               [z] sampled data information array, 3 collumns: 
%               point index, the cluster it was sampled from and it's label
%               
%               [eu] error accosiated with propogating the majority label
%               to unlabelled instances, for each node
%               
%               [wu] node weights - proportion of the total data in each
%               node
%
%
% outputs:      [P_] output (refined) pruning
%
%               [L] the majority label for each cluster in the refined
%               pruning
%
%               [XL] is the labelled dataset provided by the DH learner, 
%               including propogated and queried labels. Rows represent 
%               observations. column 1 is the observation index (in X), 
%               column 2 is the class label.

L = zeros(1,length(u)); % admissible cluster label pairs 

P_ = P; % define working pruning
for i =1:numel(P) % for v in P (nodes in the current pruning)
    
    v = P(i); % define working node
    
    % LABEL parent node incase descendants are not admissible
    % DON'T label if: 
    % 1. leaner is in the root node, 
    % 2. if the number of addmissible nodes is < the number of obv. labels,
    % 3. there is > 1 majority label
    Lu{1} = 0; % arbituary label of root
    if numel(P)~=1 && length(Lu{v})==1 %&& numel(P)>length(unique(z(:,end))) 
       L(v) = Lu{v}; 
    end
    
    % identify frist descendants...
    chv = ch(:,v); % 1st gen children, columns are siblings
    
    % check if next gen are admissible
    Pv = [v]; %#ok<NBRAK> % init sub-pruning as the parent.
    Achv = Au(chv); % logical admissible aray for child nodes
    
    % while at least 1 pair of siblings is admissible, refine and label Pv
    while sum(sum(Achv)==2) >= 1        
        i_ch = find(sum(Achv)==2); % find indices of admissible replacment nodes
        
        % add for loop: FOR i in i_ch
        for idx = 1:numel(i_ch)
            ich = i_ch(idx);
            % error parent
            ep = eu(Pv(ich));
            % error ch 
            ech = (1/(sum(wu(chv(:,ich)))))*sum(wu(chv(:,ich)).*eu(chv(:,ich)));
            Lch = Lu(chv(:,ich)); % cell of child labels
            Lch_log = cellfun(@(x) length(x)==1, Lch); % logical to check if ch1/ch2 have 1 majority label
            if length(Lu{Pv(ich)})==1 && ech<ep && sum(Lch_log)==2
                Pv(ich) = 0; % erase the old node with a zero (maintain indicies)
                u_ = chv(:,ich);
                Pv = [Pv, reshape(u_,1,[])]; %#ok<AGROW>
            end
        end       
        Pv(Pv==0)=[]; % remove zero nodes (replaced parent clusters)
        
        if length(unique(Pv)) > size(chv,2) % if the pruning has been refined, update...
            chv = ch(:,Pv); % find next gen (n+1th gen) of nodes and re-assign
            Achv = Au(chv); % admissibility array for next gen
        else
            break % if the pruning hasn't been modified break the while loop
        end      
    end    
    % pruning of the subtree (select desc) is labelled while minimising eu
    % redefine pruning and labelling of node v (P' L' of subtree Tv)
    if length(Pv)>1
        fprintf('\nTOTAL CLUSTERS %d: nodes %s replace node [%d]',...
            length(P_), mat2str(Pv), v);
        % find location of the parent node in the working pruning
        P_(P_==P(i)) = []; % remove node v from current pruning
        P_ = [P_, Pv]; %#ok<AGROW> replace with new pruning of Tv (Pv)
        for i2 = 1:numel(Pv) % label each node of subtree
            uw = Pv(i2);
            L(uw) = Lu{uw};
        end 
    end   
end
% label
xl = zeros(N,1);
for idx = 1:numel(P)
    if numel(P)>=length(unique(z(:,end)))
        v = P(idx);
        xi = u{v};
        xl(xi) = L(v);
    end
end
% overwrite definite queries
for idx = 1:size(z,1)
    z_ = z(idx,1);
    zl = z(idx,end);
    xl(z_) = zl;
end
% labelled dataset
XL = [];
for idx = 1:N
    if xl(idx)~=0
        XL = [XL; idx, xl(idx)]; %#ok<AGROW>
    end
end
% P_ is now the refined pruning 
% NOTE: output 'P_' MUST BE ASSIGNED 'P' within the DH function, to re-assign P
end