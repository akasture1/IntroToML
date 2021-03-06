% ak7213@ic.ac.uk
% Assignment 4 - Question 4(a) 
% Large Margin SVM with Gaussian RBF on raw Handwritten Data

clc
clear
close all

rng(1)
standardize = true;
showPlots = true;
logLevel = 1;
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

%% (Simple) Perceptron Results
maxIter = 1e5;
wOpt = perceptron(X', y, maxIter);
errChecks = s' .* (wOpt' * [ones(1,m);R']);
percepTestErr = sum(errChecks < 0)/m;
fprintf('Perceptron Test Error: %3.6f\n', percepTestErr);

%% Method 1: Manually Tuning KernelScale (gamma) parameter
%gammas = 10.^(-5:5);
gammas = 10.^(-5:0.1:-2);

cvErrs = zeros(length(gammas),1);
estTestErrs = zeros(length(gammas),1);
trueTestErrs = zeros(length(gammas),1);

h = waitbar(0,'Please Wait...');
for j = 1:length(gammas)
    gamma = gammas(j);
    svmObj = fitcsvm(X,y,'Standardize',standardize, 'KernelFunction','rbf', 'KernelScale', 1/sqrt(gamma), 'BoxConstraint', Inf);
    
    % Obtain Cross-Validation Error
    cvObj = crossval(svmObj);
    cvErrs(j) = kfoldLoss(cvObj);
    
    % Obtain an Est. for the Test Error using #Support Vectors
    [numSV,~] = size(svmObj.SupportVectors);
    estTestErrs(j) = numSV/(n+1);
    
    % Obtain (true) Test Error
    labels = predict(svmObj,R);
    trueTestErrs(j) = sum(labels.*s<0)/m;
    waitbar(j/length(gammas));
end
close(h);
% cvErrs, trueTestErrs vs KernelScale
figure
semilogx(gammas,cvErrs,'Linewidth', 2);
hold on
semilogx(gammas,trueTestErrs,'Linewidth', 2);

title('Cross-Validation Error and Test Error vs Gamma','FontSize',46);
xlabel('Gamma','FontSize',36);
ylabel('Error','FontSize',36);
legend('Cross-Validation Error','True Test Error');
grid on
grid minor
set(gca,'fontsize',32);

% estTestErrs, trueTestErrs vs KernelScale
figure
semilogx(gammas,estTestErrs,'Linewidth', 2);
hold on
semilogx(gammas,trueTestErrs,'Linewidth', 2);

title('Test Error Upper-Bound Est. vs True Test Error for different KernelScale values','FontSize',46);
xlabel('KernelScale','FontSize',36);
ylabel('Error','FontSize',36);
legend('Test Error Upper-Bound Est.','True Test Error');
grid on
grid minor
set(gca,'fontsize',32);

% Print Results To Console
fprintf('Standardization: %d\n\n',standardize);

[cvErrsMin, cvErrsMinIndex] = min(cvErrs);
fprintf('-------------------METHOD #1-------------------\n');
fprintf('Min Cross-Validation Error: %3.6f\n',cvErrsMin);
fprintf('Best Gamma: %3.6f\n',gammas(cvErrsMinIndex));
fprintf('True Test-Error: %3.6f\n',trueTestErrs(cvErrsMinIndex));

%% Method 2: Use OptimizeHyperparameters option to select best KernelScale value
% Optimise SVM parameters
optOptions = struct('AcquisitionFunctionName','expected-improvement-plus','Kfold', 10,...
                    'ShowPlots', showPlots, 'Verbose', logLevel);
                
svmObj = fitcsvm(X,y,'Standardize',standardize,'CacheSize', 'maximal',...
                     'KernelFunction','rbf','BoxConstraint', Inf,...
                     'OptimizeHyperparameters',{'KernelScale'},...
                     'HyperparameterOptimizationOptions', optOptions);

% Obtain Cross-Validation Error
cvObj = crossval(svmObj);
cvErr = kfoldLoss(cvObj);

% Obtain an Est. for the Test Error using #Support Vectors
[numSV,~] = size(svmObj.SupportVectors);
estTestErr = numSV/(n+1);

% Obtain (true) Test Error
labels = predict(svmObj,R);
trueTestErr = sum(labels.*s<0)/m;

fprintf('-------------------METHOD #2-------------------\n');
fprintf('Cross-Validation Error: %3.6f\n',cvErr);
fprintf('Optimal KernelScale: %3.6f\n',svmObj.ModelParameters.KernelScale);
fprintf('True Test-Error: %3.6f\n',trueTestErr);

