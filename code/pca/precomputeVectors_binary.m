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
y(y==1)=2;
y(y==3)=2;
y(y==4)=1;

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
    [~,~,V] = svds(XTr,N);
    
    % store
    PCA_V(k,:,:) = V;
%     PCA_D(k,:,:) = D;
end

%% multiply out
XTr_binary = cell(5,1);
XTe_binary = cell(5,1);
for k = 1:K
    fprintf('\nFold number: %d', k);
    XTr = X(Indices~=k,:);
    XTe = X(Indices==k,:);
    V = squeeze(PCA_V(k,:,:));
    XTr_binary{k}= XTr*V;   % if N = 1, need to transpose V!
    XTe_binary{k} = XTe*V;
end


%% calculate output
yTr_binary = cell(5,1);
yTe_binary = cell(5,1);
for k = 1:K
    fprintf('\nFold number: %d', k);
    yTr_binary{k}= y(Indices~=k,:);
    yTe_binary{k} = y(Indices==k,:);
end

%% save into .mat file
save('XTr_binary.mat','XTr_binary')
save('XTe_binary.mat','XTe_binary')
save('yTr_binary.mat','yTr_binary')
save('yTe_binary.mat','yTe_binary')