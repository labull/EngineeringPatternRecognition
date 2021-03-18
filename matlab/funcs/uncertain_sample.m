function [q_idx] = uncertain_sample(py_xD, px_D, x_u, B, qn)
% uncertainty sampling function
% Code written by Lawrence Bull
    % DEFINE UNCERTAIN POINTS
    % highest post. entropy
    hi = [];
    for i = 1:size(x_u,1)
        hi = [hi; i, -sum(py_xD(i,:).*log(py_xD(i,:)))]; %#ok<AGROW>
    end
    hi = sortrows(hi(1:end-B,:), 2); % of the PAST data (not new batch)
    % lowest posterior predictive
    pp = [[1:size(px_D,1)]', px_D]; %#ok<NBRAK>
    pp = sortrows(pp(end+1-B:end,:), 2); % of the NEW data    

    % SELECT DATA TO SAMPLE
    % entropy
    if isempty(hi) % if there is no past unlabelled data
        qh_idx = [];
    else
        qh_idx = hi(end-qn/2+1:end,1); % query index for past data (most mixed.)
    end
    % post. pred.
    qpp_idx = pp(1:qn/2, 1);
%     qpp_idx = [];
    % combined indices (relative 
    q_idx = union(qh_idx, qpp_idx);
end
