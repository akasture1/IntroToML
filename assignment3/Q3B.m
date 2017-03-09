% Author: Anand Kasture (ak7213)
% March 2017

% EE3-23 Machine Learning Assignment 3
% Q3B - Linear Regression Baseline

clc
clear 
close all

%Config
normConfig = '';
lambdaConfig = 'valAllUsers';

% Load Train/Test Data (userId, movieId, rating)
trainData = csvread('./movie-data/ratings-train.csv',1);
testData = csvread('./movie-data/ratings-test.csv',1);

% Load Features (movieId, feat1, ..., feat18)
features  = csvread('./movie-data/movie-features.csv',1);
[numMovies, numFeat] = size(features(:,2:end));

% Normalisation

if strcmpi(normConfig, 'normc')
    features(:,2:end) = (normc(features(:,2:end)'))';
    trainData(:,3) = normc(trainData(:,3));
    testData(:,3) = normc(testData(:,3));
elseif strcmpi(normConfig, 'zscore')
    features(:,2:end) = (zscore(features(:,2:end)'))';   
    trainData(:,3) = zscore(trainData(:,3));
    testData(:,3) = zscore(testData(:,3));
end        

userIds = unique(trainData(:,1));
weights = zeros(numFeat+1,length(userIds));
lambdas = zeros(length(userIds),1);

trainError = zeros(length(userIds),1);
testError = zeros(length(userIds),1);

lambda = 1;
if strcmpi(lambdaConfig, 'valAllUsers')
    [minIndex, lambdas, lambdasValErrors, lambdasTrainErrors] = getLambdaAllUsers(trainData,features,10);
    lambda = lambdas(minIndex);
end

% Obtain Weights for every user in the Training Set
for userId = 1:1:length(userIds)
    currTrainData = trainData(trainData(:,1)==userId,:);
    currTestData = testData(testData(:,1)==userId,:);
    
    % Extract training movie ratings and the corresponding features for current user
    y = currTrainData(:,3);
    Z = features(currTrainData(:,2),:); 
    Z(:,1) = 1;
    
    if strcmpi(lambdaConfig, 'valPerUser')
        lambda = getLambdaPerUser(currTrainData,Z,10);
        lambdas(userId) = lambda;
    end
    
    % Obtain weights for current user
    wReg = ridgeRegression(Z,y,lambda);
    weights(:,userId) = wReg;
    
    % Calculate Training Error
    trainError(userId) = sum((y - Z*wReg).^2);
    
    % Calculate Test Error
    y = currTestData(:,3);
    Z = features(currTestData(:,2),:); 
    Z(:,1) = 1;
    testError(userId) = sum((y - Z*wReg).^2);
end

trainError = sum(trainError)/length(trainData);
testError = sum(testError)/length(testData);

% Print Results to Console
fprintf('Training Error: %f\nTest Error: %f\n',trainError, testError);

% Plots
if strcmpi(lambdaConfig, 'valAllUsers')
    figure
    plot(lambdas,lambdasValErrors, 'x', 'MarkerSize', 15, 'LineWidth', 2);
    hold on
    plot(lambdas,lambdasTrainErrors, 'x', 'MarkerSize', 15, 'LineWidth', 2);
    
    % Figure options
    title('Cross-Validation Results for Different Lambda Values','FontSize',46);
    xlabel('Lambda','FontSize',36);
    ylabel('Mean Square Error','FontSize',36);
    legend('Cross-Validation Error','Training Error','Location','southeast');
    grid on
    grid minor
    set(gca,'fontsize',32);
elseif strcmpi(lambdaConfig, 'valPerUser')
    figure
    h = histogram(lambdas);
    h.NumBins = 20;
end

