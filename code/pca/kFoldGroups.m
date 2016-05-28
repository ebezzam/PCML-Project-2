function [ groups ] = kFoldGroups( y, K, nSeed )
%KFOLDGROUPS Create K groups with same number of each class

    rng(nSeed);
    groups = ones(size(y)); % group assignment for each sample
    nClass = unique(y);
    for c = 1:length(nClass)
        idx_c = find(y==nClass(c));
        N_c = length(idx_c);
        idx = randperm(N_c);
        % split evenly for each group
        for k = 0:K-1
            vals = (1+k*floor(N_c/K)):(k+1)*floor(N_c/K);
            groups(idx_c(idx(vals))) = k+1;
        end
    end
    

end

