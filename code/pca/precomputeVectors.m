%% store PCA vectors for K = 5 folds
clear all; close all;

%% load data
K = 5; N = 400;
load('../data/train.mat');
addpath(genpath('/Applications/MATLAB_R2014b.app/toolbox/stats/stats'));
nSeed = 8339;
rng(nSeed);
X = [double(train.X_hog), double(train.X_cnn)]; % rank of 5998
y = double(train.y);

%% k-fold
Indices = kFoldGroups(y, K, nSeed);
fprintf('\nPerforming k-folds...');
rng(nSeed);
PCA_V = ones(K,size(X,2),N);
PCA_D = ones(K,N,N);
%%
for k = 1:K
    fprintf('\nFold number: %d', k);
    XTr = X(Indices~=k,:);
    XTe = X(Indices==k,:);

    % calculate prominent singular values
    fprintf('\nCalculating SVD...\n');
    [~,D,V] = svds(XTr,N);
    
    % store
    PCA_V(k,:,:) = V;
    PCA_D(k,:,:) = D;
end

%% multiply out
XTr_r = cell(5,1);
XTe_r = cell(5,1);
for k = 1:K
    fprintf('\nFold number: %d', k);
    XTr = X(Indices~=k,:);
    XTe = X(Indices==k,:);
    V = squeeze(PCA_V(k,:,:));
    XTr_r{k}= XTr*V;    % if N = 1, need to transpose V!
    XTe_r{k} = XTe*V;
end


%% calculate output
yTr = cell(5,1);
yTe = cell(5,1);
for k = 1:K
    fprintf('\nFold number: %d', k);
    yTr{k}= y(Indices~=k,:);
    yTe{k} = y(Indices==k,:);
end

%% save into .mat file
save('XTr_r.mat','XTr_r')
save('XTe_r.mat','XTe_r')
save('yTr.mat','yTr')
save('yTe.mat','yTe')