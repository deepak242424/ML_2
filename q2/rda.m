% Get the dataset
T = readtable('iris_data.txt');
X1 = T(:,3:4);
Y = T(:,5);
X = normc(table2array(X1));
labels = table2array(Y);
[y, support] = canonizeLabels(labels);
model = discrimAnalysisFit(X, y, 'quadratic');

h = plotDecisionBoundary(X, y, @(Xtest)discrimAnalysisPredict(model, Xtest));
title(sprintf('Discrim. analysis of QDA'));
xlabel('X_1'); ylabel('X_2');
