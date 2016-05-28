function [ errTr, errTe ] = optimizeNN(type, K, dimVals, valsEpoch, valsHiddenSize, valsBatch, nSeed)
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
    errTr = zeros(length(dimVals), length(valsEpoch), length(valsHiddenSize), length(valsBatch), K);
    errTe = zeros(length(dimVals), length(valsEpoch), length(valsHiddenSize), length(valsBatch), K);
    
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
        
        % grid search over possible hyper-parameters
        fprintf('\nPerforming grid search...\n');
        for d = 1:length(dimVals)
            % reduce dimensions
            dim = dimVals(d);
            fprintf('\nDimension: %d', dim);
            XTr_r1 = single(X_tr(:,1:dim));
            XTe_r1 = single(X_te(:,1:dim));
            % setup
            alpha = 5;
            opts.plot = 0;                  % plot training error
            nn.learningRate = alpha;
            for b = 1:length(valsEpoch)
                fprintf('\nNumber of epochs: %d', valsEpoch(b));
                opts.numepochs = valsEpoch(b);           %  Number of full sweeps through data
                for h = 1:length(valsHiddenSize)
                    Nh = valsHiddenSize(h);
                    fprintf('\nNumber of hidden nodes: %d', Nh);
                    for q = 1:length(valsBatch)
                        nn = nnsetup([size(XTr_r1,2) Nh max(double(y_tr))]);
                        batchSize = valsBatch(q);
                        fprintf('\nMini-batch size: %d', batchSize);
                        opts.batchsize = batchSize;     %  Take a mean gradient step over this many samples
                        % num of samples must be multiple of batch size
                        numSampToUse = opts.batchsize * floor( size(XTr_r1) / opts.batchsize);
                        XTr = XTr_r1(1:numSampToUse,:);
                        yTr_r = y_tr(1:numSampToUse);
                        % normalize data
                        [XTr_n, mu, sigma] = zscore(XTr); % train, get mu and std
                        XTe_n = normalize(XTe_r1, mu, sigma);  % normalize test data
                        % prepare labels for NN
                        if type == 1
                            LL = [1*(yTr_r == 1), 1*(yTr_r == 2), 1*(yTr_r == 3), 1*(yTr_r == 4) ];
                        else
                            LL = [1*(yTr_r == 1), 1*(yTr_r == 2) ];
                        end
                        % train model
                        [nn, L] = nntrain(nn, XTr_n, LL, opts);
                        % obtain prediction for training set
                        nn.testing = 1;
                        nn = nnff(nn, XTr_n, zeros(size(XTr_n,1), nn.size(end)));
                        nn.testing = 0;
                        nnPred = nn.a{end};
                        [~,predTr] = max(nnPred,[],2);
                        % obtain prediction for test set
                        nn.testing = 1;
                        nn = nnff(nn, XTe_n, zeros(size(XTe_n,1), nn.size(end)));
                        nn.testing = 0;
                        nnPred = nn.a{end};
                        [~,predTe] = max(nnPred,[],2);
                        % compute BER
                        [ errTr(d,b,h,q,k), ~ ] = compute_BER(double(yTr_r), double(predTr), double(max(yTr_r)));
                        [ errTe(d,b,h,q,k), ~ ] = compute_BER(double(y_te), double(predTe), double(max(yTr_r)));
                        clearvars nn
                    end
                end
            end
        end
    end

end

