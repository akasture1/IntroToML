% Author: Anand Kasture (ak7213)
% March 2017

% EE3-23 Machine Learning Assignment 3
% Q3D - Collaborative Filtering
clc
clear
close all

%% Setup
% Feature Vector Dimension
K = 15;

% Regulariser Constant
lambda = 0.001;

% Stochastic Gradient Descent (SGD) Step Size
gamma = 0.01;

% Max Epochs
epoch = 1;
maxEpochs = 100;

% Load Train/Test Data (userId, movieId, rating)
trainData = csvread('./movie-data/ratings-train.csv',1);
testData = csvread('./movie-data/ratings-test.csv',1);

[numTrainEntries, ~] = size(trainData);
[numTestEntries, ~] = size(testData);

uniqueTrainUsers = unique(trainData(:,1));
numUniqueTrainUsers = length(uniqueTrainUsers);

% Load Features (movieId, feat1, ..., feat18)
features  = csvread('./movie-data/movie-features.csv',1);
[numMovies, ~] = size(features);

% Train/Test Error Vectors
trainErrors = zeros(maxEpochs,1);
testErrors = zeros(maxEpochs,1);

% Useful SGD Functions
getError = @(r, u, v) r - (u*v');
nextUserFeat = @(r, u, v, gamma, lambda) u + gamma*(getError(r,u,v)*v - lambda*u);
nextMovieFeat = @(r, u, v, gamma, lambda) v + gamma*(getError(r,u,v)*u - lambda*v);

% Initialise RANDOM feature vectors
userFeatures = rand(numUniqueTrainUsers, K);
movieFeatures = rand(numMovies, K);

%% Iterate!
fprintf('Starting Stochastic Gradient Descent Iterations\n');
while epoch <= maxEpochs
    fprintf('Starting epoch %d\n',epoch)
    for j = 1:1:numTrainEntries
        trainEntry = trainData(j,:);
        r = trainEntry(3);
        u = userFeatures(trainEntry(1),:);
        v = movieFeatures(trainEntry(2),:);
        
        userFeatures(trainEntry(1),:) = nextUserFeat(r, u, v, gamma, lambda);
        movieFeatures(trainEntry(2),:) = nextMovieFeat(r, u, v, gamma, lambda);
    end    
    
    trainError = 0; testError = 0;
    for user = 1:1:numUniqueTrainUsers
        currTrainData = trainData(trainData(:,1)==user,:);
        currTestData = testData(testData(:,1)==user,:);
        U = userFeatures(user,:);
        
        % Calculate Training Error
        R = currTrainData(:,3);
        V = movieFeatures(currTrainData(:,2),:); 
        trainError = trainError + sum((R - V*U').^2);
        
        % Calculate Test Error
        R = currTestData(:,3);
        V = movieFeatures(currTestData(:,2),:); 
        testError = testError + sum((R - V*U').^2);
    end
    trainErrors(epoch) = (1/numTrainEntries)*trainError;
    testErrors(epoch) = (1/numTestEntries)*testError;
    fprintf('Training Error: %3.6f; Test Error: %3.6f\n',trainErrors(epoch), testErrors(epoch));
    epoch = epoch + 1;
end

%% DONE!

% Plot Error Results
figure
plot(1:1:maxEpochs,trainErrors, '-', 'MarkerSize', 15, 'LineWidth', 2);
hold on
plot(1:1:maxEpochs,testErrors, '-', 'MarkerSize', 15, 'LineWidth', 2);

% Figure options
title('Collaborative Filtering: Training vs Test Error','FontSize',46);
xlabel('Epochs','FontSize',36);
ylabel('Error','FontSize',36);
legend('Training Error','Test Error','Location','southeast');
grid on
grid minor
set(gca,'fontsize',32);



