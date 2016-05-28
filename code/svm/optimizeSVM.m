function [ errTr, errTe ] = optimizeSVM(K, dimVals, regVals, kernelType, nSeed)
%KFOLDS 

    load('../pca/XTr_binary.mat');
    load('../pca/XTe_binary.mat');
    load('../pca/yTr_binary.mat');
    load('../pca/yTe_binary.mat');

    % initizalize error
    errTr = zeros(length(dimVals), length(regVals), K);
    errTe = zeros(length(dimVals), length(regVals), K);
    
    % k-folds
    fprintf('\nPerforming k-folds...');
    rng(nSeed);
    for k = 1:K
        % obtain corresponding pre-computed train and test for fold k
        fprintf('\n\nFold number: %d', k);
        X_tr = XTr_r{k};
        X_te = XTe_r{k};
        y_tr = yTr{k};
        y_te = yTe{k};
        
        % change to {-1,1} for svm
        y_tr(y_tr==2)=-1;
        y_te(y_te==2)=-1;
        
        % grid search over possible hyper-parameters
        fprintf('\nPerforming grid search...');
        for d = 1:length(dimVals)
            fprintf('\nDimension size: %d', dimVals(d));
            % reduce dimensions
            dim = dimVals(d);
            X_tr_r = X_tr(:,1:dim);
            X_te_r = X_te(:,1:dim);
            for n = 1:length(regVals)
                % learn RF model 
                fprintf('\nValue for C: %d', regVals(n));
                if(strcmp(kernelType,'polynomial'))
                    SVMModel = fitcsvm(X_tr_r,y_tr, 'BoxConstraint', regVals(n), 'KernelFunction', kernelType, 'PolynomialOrder',2);
                else
                    SVMModel = fitcsvm(X_tr_r,y_tr, 'BoxConstraint', regVals(n), 'KernelFunction', kernelType);
                end
                % apply model
                predTr = predict(SVMModel,X_tr_r);
                predTe = predict(SVMModel,X_te_r);
                % train error
                predTr(predTr==-1)=2;
                [ errTr(d,n,k), ~ ] = compute_BER(y_tr, predTr, max(y_tr));
                % test error
                predTe(predTe==-1)=2;
                [ errTe(d,n,k), ~ ] = compute_BER(y_te, predTe, max(y_tr));
            end
        end
    end


end

