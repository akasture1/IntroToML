% ak7213@ic.ac.uk
% Assignment 4 - Question 4(c) pt.2
% 2-D Feature Vectors Plot

clc
clear
close all

rng(1)
standardize = true;
showPlots = true;
logLevel = 1;

%% Extract Training and Test Data for Digits 1 and 0,2,...,9
% Feature Training Data
X = importdata('./data/features.train');
y = X(:,1);

% Map digit -> class : {1,Others} -> {-1,1}
y(y(:)==1) = -1;
y(y(:)~=-1) = 1;
X = X(:,2:end);

[n,~] = size(X);

% Feature Test Data
R = importdata('./data/features.test');
s = R(:,1);

% Map digit -> class : {1,Others} -> {-1,1}
s(s(:)==1) = -1;
s(s(:)~=-1) = 1;
R = R(:,2:end);

[m,~] = size(R);

%% Train SVM and measure performance

optOptions = struct('AcquisitionFunctionName','expected-improvement-plus','Kfold', 10,...
                    'ShowPlots', showPlots, 'Verbose', logLevel);
% Optimise SVM parameters               
svmObj = fitcsvm(X,y,'Standardize',standardize,'CacheSize', 'maximal',...
                     'KernelFunction','rbf',...
                     'OptimizeHyperparameters', 'auto',...
                     'HyperparameterOptimizationOptions', optOptions);

% Record optimal SVM parameters
kernelScale = svmObj.ModelParameters.KernelScale;
boxConstraint = svmObj.ModelParameters.BoxConstraint;

% Obtain Cross-Validation Error
cvObj = crossval(svmObj);
cvErr = kfoldLoss(cvObj);

% Obtain an Est. for the Test Error using #Support Vectors
[numSV,~] = size(svmObj.SupportVectors);
estTestErr = numSV/(n+1);

% Obtain Training Error
labels = predict(svmObj,X);
trainErr = sum(labels.*y<0)/n;

% Obtain (true) Test Error
labels = predict(svmObj,R);
trueTestErr = sum(labels.*s<0)/m;

%% Generate 2-D Feature Vectors Plot
figure
d=0.01;
[x1Grid, x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)), min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
[~,scores] = predict(svmObj, xGrid);

plot(X((y==-1),1), X((y==-1),2), 'x', 'MarkerSize', 6, 'LineWidth', 1.2);
hold on
plot(X((y==1),1), X((y==1),2), 'o', 'MarkerSize', 6, 'LineWidth', 1.2);
plot(X(svmObj.IsSupportVector,1),X(svmObj.IsSupportVector,2),'ko','MarkerSize',7,'LineWidth', 2)
contour(x1Grid,x2Grid, reshape(scores(:,2), size(x1Grid)), [0,0], 'LineWidth', 2);

title('Handwritten Digit Classification: 2-D Feature Vectors','FontSize',46);
xlabel('x_{1} (Intensity)','FontSize',36);
ylabel('x_{2} (Symmetry)','FontSize',36);
legend('Digit 1', 'Other Digits', 'Support Vectors', 'Decision Boundary');
set(gca,'fontsize',32);

