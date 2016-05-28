% Author: Eric Bezzam, Date: 12/12/2015
clear all; clc; close all;

%% K-fold cross validation
K = 5;
nSeed = 8339;
rng(nSeed);
dimVals = [25:25:200,200:100:400];
regVals = 0.5:0.5:1.5;      % box constraint
kernelType = 'linear'; % linear, polynomial, rbf
[ errTr, errTe] = optimizeSVM(K, dimVals, regVals, kernelType, nSeed);

%% plot results
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
    std_Te(n) = std(errTe(n,idx(n),:));
end
plot(dimVals, minDimTr, '-b', 'LineWidth',3);
plot(dimVals, minDimTe, '-r', 'LineWidth',3);
jbfill(dimVals', minDimTe + std_Te, minDimTe - std_Te, [1.0,0.8,0.8]); hold on;
plot(dimVals, minDimTr, '-b', 'LineWidth',3);
plot(dimVals, minDimTe, '-r', 'LineWidth',3);
legend('training', 'test', 'standard deviation', 'Location', 'Best')
xlim([min(dimVals), max(dimVals)])
title('SVM Cross-Validation')
hx = xlabel('Number of features');
hy = ylabel('BER');
% ylim([0, 0.4])
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off
[minTestErr, dimIdx] = min(minDimTe);
%% box constraint vs BER
figure(2)
hold on
[minDimTe, idx] = min(meanTest,[],1);
minDimTr = minDimTe;
for n = 1: length(regVals)
    minDimTr(n) = meanTrain(idx(n),n);
end
% calculate standard deviation
std_Te = minDimTr;
for n = 1: length(regVals)
    std_Te(n) = std(errTe(idx(n),n,:));
end
plot(regVals, minDimTr, '-b', 'LineWidth',3);
plot(regVals, minDimTe, '-r', 'LineWidth',3);
jbfill(regVals, minDimTe + std_Te, minDimTe - std_Te, [1.0,0.8,0.8]); hold on;
plot(regVals, minDimTr, '-b', 'LineWidth',3);
plot(regVals, minDimTe, '-r', 'LineWidth',3);
legend('training', 'test', 'standard deviation', 'Location', 'West')
xlim([min(regVals), max(regVals)])
title('SVM Cross-Validation')
hx = xlabel('Box constraint');
hy = ylabel('BER');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
hold off
[~, regIdx] = min(minDimTe);
minTestErrVar = std(errTe(dimIdx,regIdx,:));
fprintf('\n\nOptimal number of principal components: %d', dimVals(dimIdx));
fprintf('\nOptimal box constraint value: %d', regVals(regIdx));
fprintf('\nEstimated test error: %.2f+/-%.2f%%\n', minTestErr*100, minTestErrVar*100);

