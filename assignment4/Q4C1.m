% ak7213@ic.ac.uk
% Assignment 4 - Question 4(c) pt.1
% 2-D PCA Approximation Plot

clc
clear
close all

rng(1)
normalise = true;

%% Extract Training and Test Data for Digits 1 and 0,2,...,9
% Raw Training Data
X = importdata('./data/zip.train');
y = X(:,1);

% Map digit -> class : {1,Others} -> {-1,1}
y(y(:)==1) = -1;
y(y(:)~=-1) = 1;
X = X(:,2:end);

[n,~] = size(X);

% Raw Test Data
R = importdata('./data/zip.test');
s = R(:,1);

% Map digit -> class : {1,Others} -> {-1,1}
s(s(:)==1) = -1;
s(s(:)~=-1) = 1;
R = R(:,2:end);

[m,~] = size(R);

%% Construct PCA Coefficient Matrices
coeffs = pca(X);

%% Train SVM and measure performance
XBar = X*coeffs(:,1:2);
RBar = R*coeffs(:,1:2);

% Optimise SVM parameters
svmObj = fitcsvm(XBar,y,'Standardize',normalise,'KernelFunction','rbf',...
                        'OptimizeHyperparameters','auto',...
                        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
                        'expected-improvement-plus'));

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
labels = predict(svmObj,XBar);
trainErr = sum(labels.*y<0)/n;

% Obtain (true) Test Error
labels = predict(svmObj,RBar);
trueTestErr = sum(labels.*s<0)/m;

%% Generate Figure 2-D PCA Approximation Figure
figure
d=0.01;
[x1Grid, x2Grid] = meshgrid(min(XBar(:,1)):d:max(XBar(:,1)), min(XBar(:,2)):d:max(XBar(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
[~,scores] = predict(svmObj, xGrid);

plot(XBar((y==-1),1), XBar((y==-1),2), 'x', 'MarkerSize', 6, 'LineWidth', 1.2);
hold on
plot(XBar((y==1),1), XBar((y==1),2), 'o', 'MarkerSize', 6, 'LineWidth', 1.2);
plot(XBar(svmObj.IsSupportVector,1),XBar(svmObj.IsSupportVector,2),'ko','MarkerSize',7,'LineWidth', 2)
contour(x1Grid,x2Grid, reshape(scores(:,2), size(x1Grid)), [0,0], 'LineWidth', 2);

title('Handwritten Digit Classification: 2-Dimensional PCA Approximation','FontSize',46);
xlabel('x_{1}','FontSize',36);
ylabel('x_{2}','FontSize',36);
legend('Digit 1', 'Other Digits', 'Support Vectors', 'Decision Boundary');
set(gca,'fontsize',32);

