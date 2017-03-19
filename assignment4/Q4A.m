% ak7213@ic.ac.uk
% Assignment 4 - Question 4(a) 
% Large Margin SVM with Gaussian RBF on raw Handwritten Data

clear
close all

rng(1)
gamma = 1e3;

%% Extract Training and Test Data for Digits 2 and 8

% Raw Training Data
X = importdata('./data/zip.train');
X = X(X(:,1)==2 | X(:,1)==8 ,:);
y = X(:,1);

% Map digit -> class : {2,8} -> {1,-1}
y(y(:)==2) = -1;
y(y(:)==8) = 1;
X = X(:,2:end);

[n,d1] = size(X);

% Raw Test Data
% Filter for digits 2 and 8
R = importdata('./data/zip.test');
R = R(R(:,1)==2 | R(:,1)==8 ,:);
s = R(:,1);

% Map digit -> class : {2,8} -> {1,-1}
s(s(:)==2) = -1;
s(s(:)==8) = 1;
R = R(:,2:end);

[m,d2] = size(R);

%% Method 1
gammas = 10.^(-5:5);
g = length(gammas);
cvErrs = zeros(g,1);
testErrUBs = zeros(g,1);
testErrs = zeros(g,1);
for j = 1:g
    gamma = gammas(j);
    svm = fitcsvm(X,y,'Standardize',true, 'KernelFunction','rbf', 'KernelScale', gamma, 'BoxConstraint', Inf);
    cvSvm = crossval(svm);
    
    cvErrs(j) = kfoldLoss(cvSvm);
    [numSV,~] = size(svm.SupportVectors);
    testErrUBs(j) = numSV/(n+1);
    
    labels = predict(svm,R);
    testErrs(j) = numel(find(labels.*s<0))/m;
end

plot(cvErrs)
hold on
plot(testErrUBs)
plot(testErrs)
%% Method 2
