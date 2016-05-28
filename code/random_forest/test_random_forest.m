% Author: Eric Bezzam, Date: 13/12/2015
clear all; close all;


%% K-fold cross validation
addpath(genpath('../piotr_toolbox'));   % this was downloaded from the link provided in the PCML course page
addpath(genpath('../piotr_toolbox/classify'));
K = 5;
nSeed = 8339;
rng(nSeed);
dimVals = 2:2:100;
treeVals = 5:5:200;
type = 0;   % 0 for binary, 1 for multi
[ errTr, errTe] = optimizeRF(type, K, dimVals, treeVals, nSeed);


%% plot results
%% average folds
meanErrTr = mean(errTr,3);
meanErrTe = mean(errTe,3);
%% dimension vs. BER
figure(1)
hold on
[minDimTe, idx] = min(meanErrTe,[],2);
minDimTr = minDimTe;
for d = 1: length(dimVals)
    minDimTr(d) = meanErrTr(d,idx(d));
end
% calculate standard deviation
std_Te = minDimTr;
for d = 1: length(dimVals)
    std_Te(d) = std(errTe(d,idx(d),:));
end
plot(dimVals, minDimTr, '-b', 'LineWidth',3); hold on;
plot(dimVals, minDimTe, '-r', 'LineWidth',3);
jbfill(dimVals', minDimTe + std_Te, minDimTe - std_Te, [1.0,0.8,0.8]); hold on;
plot(dimVals, minDimTr, '-b', 'LineWidth',3); hold on;
plot(dimVals, minDimTe, '-r','LineWidth',3);
legend('training', 'test', 'standard deviation')
xlim([min(dimVals), max(dimVals)])
ylim([0, 0.2])
title('RF Cross-Validation')
hx = xlabel('Number of principal components');
hy = ylabel('BER');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off
%% nTrees vs. BER
figure(2)
hold on
[minDimTe, idx] = min(meanErrTe,[],1);
minDimTr = minDimTe;
for n = 1: length(treeVals)
    minDimTr(n) = meanErrTr(idx(n),n);
end
% calculate standard deviation
std_Te = minDimTr;
for n = 1: length(treeVals)
    std_Te(n) = std(errTe(idx(n),n,:));
end
plot(treeVals, minDimTr, '-b', 'LineWidth',3); hold on;
plot(treeVals, minDimTe, '-r', 'LineWidth',3);
jbfill(treeVals, minDimTe + std_Te, minDimTe - std_Te, [1.0,0.8,0.8]); hold on;
plot(treeVals, minDimTr, '-b', 'LineWidth',3); hold on;
plot(treeVals, minDimTe, '-r','LineWidth',3);
legend('training', 'test', 'standard deviation', 'Location', 'SouthEast')
xlim([min(treeVals), max(treeVals)])
% ylim([0, 0.15])
title('RF Cross-Validation')
hx = xlabel('Number of trees');
hy = ylabel('BER');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off

