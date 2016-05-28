% Author: Eric Bezzam, Date: 16/12/2015
clear all; close all;


%% K-fold cross validation
addpath(genpath('../DeepLearnToolbox-master')); %% this is the same library provided on the PCML course page
nSeed = 8339;
rng(nSeed);
K = 5;
%% apply k-fold
dimVals = 300;
valsEpoch = 20:5:40;
valsHiddenSize = 400; 
valsBatch = 10:20:90;
type = 0;   % 0 for binary, 1 for multi
[ errTr, errTe] = optimizeNN(type, K, dimVals, valsEpoch, valsHiddenSize, valsBatch, nSeed);

%% plot results
%% average over folds
meanErrTr = mean(errTr,5);
meanErrTe = mean(errTe,5);
meanTest = squeeze(meanErrTe);
meanTrain = squeeze(meanErrTr);

%% varying dimension and hidden size
% dimension vs BER
figure(1)
hold on
[minDimTe, idx] = min(meanTest,[],2);
minDimTr = minDimTe;
for n = 1: length(dimVals)
    minDimTr(n) = meanTrain(n,idx(n));
end
% calculate standard deviation
std_Te = minDimTr;
for n = 1: length(dimVals)
    std_Te(n) = std(errTe(n,1,idx(n),:));
end
plot(dimVals, minDimTr, '-b', 'LineWidth',3);
plot(dimVals, minDimTe, '-r', 'LineWidth',3);
jbfill(dimVals', minDimTe + std_Te, minDimTe - std_Te, [1.0,0.8,0.8]); hold on;
plot(dimVals, minDimTr, '-b', 'LineWidth',3);
plot(dimVals, minDimTe, '-r', 'LineWidth',3);
legend('training', 'test', 'standard deviation', 'Location', 'Best')
xlim([min(dimVals), max(dimVals)])
title('NN Cross-Validation')
hx = xlabel('Number of features');
hy = ylabel('BER');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off
[minTestErr, dimIdx] = min(minDimTe);
% hidden vs BER
figure(2)
hold on
[minDimTe, idx] = min(meanTest,[],1);
minDimTr = minDimTe;
for n = 1: length(valsHiddenSize)
    minDimTr(n) = meanTrain(idx(n),n);
end
% calculate standard deviation
std_Te = minDimTr;
for n = 1: length(valsHiddenSize)
    std_Te(n) = std(errTe(idx(n),1,n,:));
end
plot(valsHiddenSize, minDimTr, '-b', 'LineWidth',3);
plot(valsHiddenSize, minDimTe, '-r', 'LineWidth',3);
jbfill(valsHiddenSize, minDimTe + std_Te, minDimTe - std_Te, [1.0,0.8,0.8]); hold on;
plot(valsHiddenSize, minDimTr, '-b', 'LineWidth',3);
plot(valsHiddenSize, minDimTe, '-r', 'LineWidth',3);
legend('training', 'test', 'standard deviation', 'Location', 'Best')
xlim([min(valsHiddenSize), max(valsHiddenSize)])
title('NN Cross-Validation')
hx = xlabel('Number of hidden nodes');
hy = ylabel('BER');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off
[~, hiddenIdx] = min(minDimTe);
minTestErrVar = std(errTe(dimIdx,1,hiddenIdx,:));
fprintf('\n\nOptimal number of principal components: %d', dimVals(dimIdx));
fprintf('\nOptimal number of hidden layer nodes: %d', valsHiddenSize(hiddenIdx));
fprintf('\nEstimated test error: %.2f+/-%.2f%%\n', minTestErr*100, minTestErrVar*100);

%% varying epochs and batch size
% epochs vs BER
figure(1)
hold on
[minDimTe, idx] = min(meanTest,[],2);
minDimTr = minDimTe;
for n = 1: length(valsEpoch)
    minDimTr(n) = meanTrain(n,idx(n));
end
% calculate standard deviation
std_Te = minDimTr;
for n = 1: length(valsEpoch)
    std_Te(n) = std(errTe(1,n,1,idx(n),:));
end
plot(valsEpoch, minDimTr, '-b', 'LineWidth',3);
plot(valsEpoch, minDimTe, '-r', 'LineWidth',3);
jbfill(valsEpoch', minDimTe + std_Te, minDimTe - std_Te, [1.0,0.8,0.8]); hold on;
plot(valsEpoch, minDimTr, '-b', 'LineWidth',3);
plot(valsEpoch, minDimTe, '-r', 'LineWidth',3);
legend('training', 'test', 'standard deviation', 'Location', 'Best')
xlim([min(valsEpoch), max(valsEpoch)])
title('NN Cross-Validation')
hx = xlabel('Number of epochs');
hy = ylabel('BER');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off
[minTestErr, epochIdx] = min(minDimTe);
% batch size vs BER
figure(2)
hold on
[minDimTe, idx] = min(meanTest,[],1);
minDimTr = minDimTe;
for n = 1: length(valsBatch)
    minDimTr(n) = meanTrain(idx(n),n);
end
% calculate standard deviation
std_Te = minDimTr;
for n = 1: length(valsBatch)
    std_Te(n) = std(errTe(1,idx(n),1,n,:));
end
plot(valsBatch, minDimTr, '-b', 'LineWidth',3);
plot(valsBatch, minDimTe, '-r', 'LineWidth',3);
jbfill(valsBatch, minDimTe + std_Te, minDimTe - std_Te, [1.0,0.8,0.8]); hold on;
plot(valsBatch, minDimTr, '-b', 'LineWidth',3);
plot(valsBatch, minDimTe, '-r', 'LineWidth',3);
legend('training', 'test', 'standard deviation', 'Location', 'Best')
xlim([min(valsBatch), max(valsBatch)])
title('NN Cross-Validation')
hx = xlabel('Mini-batch size');
hy = ylabel('BER');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off
[~, batchIdx] = min(minDimTe);
minTestErrVar = std(errTe(1,epochIdx,1,batchIdx,:));
fprintf('\n\nOptimal number of epochs: %d', valsEpoch(epochIdx));
fprintf('\nOptimal mini-batch size: %d', valsBatch(batchIdx));
fprintf('\nEstimated test error: %.2f+/-%.2f%%\n', minTestErr*100, minTestErrVar*100);
