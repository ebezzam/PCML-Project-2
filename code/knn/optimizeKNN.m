function [ errTr, errTe ] = optimizeKNN(type, K, dimVals, neighVals, nSeed)
%KFOLDS 

    % load precomputed reductions
    if type==0  % load binary
        load('../pca/XTr_binary.mat');
        load('../pca/XTe_binary.mat');
        load('../pca/yTr_binary.mat');
        load('../pca/yTe_binary.mat');
    end
    if type==1  % load multi
        load('../pca/XTr_r.mat');
        load('../pca/XTe_r.mat');
        load('../pca/yTr.mat');
        load('../pca/yTe.mat');
    end
    
    % initizalize error
    errTr = zeros(length(dimVals), length(neighVals), K);
    errTe = zeros(length(dimVals), length(neighVals), K);
    
    % k-folds
    fprintf('\nPerforming k-folds...');
    rng(nSeed);
    for k = 1:K
        % obtain corresponding pre-computed train and test for fold k
        fprintf('\nFold number: %d', k);
        X_tr = XTr_r{k};
        X_te = XTe_r{k};
        y_tr = yTr{k};
        y_te = yTe{k};
        
        % grid search over possible hyper-parameters
        fprintf('\nPerforming grid search...\n');
        for d = 1:length(dimVals)
            fprintf('\nDimension size: %d', dimVals(d));
            % reduce dimensions
            dim = dimVals(d);
            X_tr_r = X_tr(:,1:dim);
            X_te_r = X_te(:,1:dim);
            for n = 1:length(neighVals)
                fprintf('\nNumber of neighbors: %d', neighVals(n));
                % learn KNN model
                neigh = neighVals(n);
                Mdl = fitcknn(X_tr_r,y_tr,'NumNeighbors',neigh,'Standardize',1);
                % apply model
                predTr = predict(Mdl, X_tr_r);
                predTe = predict(Mdl, X_te_r);
                % train error
                [ errTr(d,n,k), ~ ] = compute_BER(y_tr, predTr, max(y_tr));
                % test error
                [ errTe(d,n,k), ~ ] = compute_BER(y_te, predTe, max(y_tr));
            end
        end
    end


end

