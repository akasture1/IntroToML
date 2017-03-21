% ak7213@ic.ac.uk
% Assignment 4 - Question 4(b) 
% Soft Margin SVM with Gaussian RBF on raw Handwritten Data
% Determining Optimal PCA dimension 

clc
clear
close all

rng(1)
standardize = true;
showPlots = false;
logLevel = 0;

%% Extract Training and Test Data for Digits 2 and 8
% Raw Training Data
X = importdata('./data/zip.train');
X = X(X(:,1)==2 | X(:,1)==8 ,:);
y = X(:,1);

% Map digit -> class : {2,8} -> {-1,1}
y(y(:)==2) = -1;
y(y(:)==8) = 1;
X = X(:,2:end);

[n,d] = size(X);

% Raw Test Data
R = importdata('./data/zip.test');
R = R(R(:,1)==2 | R(:,1)==8 ,:);
s = R(:,1);

% Map digit -> class : {2,8} -> {-1,1}
s(s(:)==2) = -1;
s(s(:)==8) = 1;
R = R(:,2:end);

[m,~] = size(R);

%% Construct PCA Coefficient Matrix
coeffs = pca(X);

%% Train SVM and measure performance
% Error Vectors
cvErrs = zeros(d,1);
estTestErrs = zeros(d,1);
trainErrs = zeros(d,1);
trueTestErrs = zeros(d,1);

% Track KernelScale and BoxConstraint variables
kernelScales = zeros(d,1);
boxConstraints = zeros(d,1);

optOptions = struct('AcquisitionFunctionName','expected-improvement-plus','Kfold', 10,...
                    'ShowPlots', showPlots, 'Verbose', logLevel);
                
% Finding best k-Dimension PCA implementation
fprintf('completed iteration:\n');
parfor k = 1:1:d
    % Reduce Dimension
    XBar = X*coeffs(:,1:k);
    RBar = R*coeffs(:,1:k);

    % Optimise SVM parameters               
    svmObj = fitcsvm(XBar,y,'Standardize',standardize,'CacheSize', 'maximal',...
                            'KernelFunction','rbf',...
                            'OptimizeHyperparameters', 'auto',...
                            'HyperparameterOptimizationOptions', optOptions);
    
    kernelScales(k) = svmObj.ModelParameters.KernelScale;
    boxConstraints(k) = svmObj.ModelParameters.BoxConstraint;
    
    % Obtain Cross-Validation Error
    cvObj = crossval(svmObj);
    cvErrs(k) = kfoldLoss(cvObj);
    
    % Obtain an Est. for the Test Error using #Support Vectors
    [numSV,~] = size(svmObj.SupportVectors);
    estTestErrs(k) = numSV/(n+1);
    
    % Obtain Training Error
    labels = predict(svmObj,XBar);
    trainErrs(k) = sum(labels.*y<0)/n;

    % Obtain (true) Test Error
    labels = predict(svmObj,RBar);
    trueTestErrs(k) = sum(labels.*s<0)/m;
    
    fprintf('%d',k);
end
