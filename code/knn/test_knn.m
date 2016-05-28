% Author: Eric Bezzam, Date: 12/12/2015
clear all; close all;

%% K-fold cross validation
addpath(genpath('/Applications/MATLAB_R2014b.app/toolbox/stats/stats'));
nSeed = 8339;
K = 5;
dimVals = [1:20,25:5:50,60:10:100];
Kmax = 75;
neighVals = 1:2:Kmax;
type = 0;   % 0 for binary, 1 for multi
[ errTr, errTe] = optimizeKNN(type, K, dimVals, neighVals, nSeed);

%% plot results
%% average over folds
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
std_Tr = minDimTr;
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
ylim([0.05, 0.18])
title('k-NN Cross-Validation')
hx = xlabel('Number of principal components');
hy = ylabel('BER');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off
%% number of neighbors vs. BER
figure(2)
hold on
meanErrTr = mean(errTr,3);
meanErrTe = mean(errTe,3);
[minDimTe, idx] = min(meanErrTe,[],1);
minDimTr = minDimTe;
for n = 1:length(neighVals)
    minDimTr(n) = meanErrTr(idx(n),n);
end
for n = 1:length(neighVals)
    std_Te(n) = std(errTe(idx(n),n,:));
end
plot(neighVals, minDimTr, '-b', 'LineWidth',3)
plot(neighVals, minDimTe, '-r', 'LineWidth',3)
jbfill(neighVals, minDimTe + std_Te', minDimTe - std_Te', [1.0,0.8,0.8]); hold on;
plot(neighVals, minDimTr, '-b', 'LineWidth',3)
plot(neighVals, minDimTe, '-r', 'LineWidth',3)
xlim([min(neighVals), max(neighVals)])
ylim([0.05, 0.15])
legend('training', 'test', 'standard deviation', 'Location', 'Best')
title('k-NN Cross-Validation')
hx = xlabel('Number of neighbors');
hy = ylabel('BER');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off
