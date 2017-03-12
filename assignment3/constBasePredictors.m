% Author: Anand Kasture (ak7213)
% March 2017

% EE3-23 Machine Learning Assignment 3
% Q3A - Constant Base Predictors

clc
clear all
close all

% Load Train/Test Data (userId, movieId, rating)
trainData = csvread('./movie-data/ratings-train.csv',1);
testData =  csvread('./movie-data/ratings-test.csv',1);

% User Rating Prediction (Movie-Independent)
trainMseU = 0; testMseU = 0;
trainNum = 0;  testNum = 0;
nUsers = max(trainData(:,1));
userRatings = [ (1:1:nUsers)', zeros(nUsers,1)] ;

for j = 1:nUsers
    trainRatings = trainData(trainData(:,1)==j,3);
    testRatings = testData(testData(:,1)==j,3);
    userRatings(j,2) = mean(trainRatings);

    trainMseU = trainMseU + sum((trainRatings-userRatings(j,2)).^2);
    trainNum = trainNum + length(trainRatings);

    testMseU = testMseU + sum((testRatings-userRatings(j,2)).^2);
    testNum = testNum + length(testRatings);
end
trainMseU = trainMseU/trainNum;
testMseU = testMseU/testNum;

fprintf('User Rating Prediction (Movie-Independent)\n');
fprintf('Training Error: %f\nTest Error: %f\n\n',trainMseU, testMseU);

% Movie Rating Prediction (User-Independent)
trainMseM = 0; testMseM = 0;
trainNum = 0;  testNum = 0;
nMovies = max(trainData(:,2));
movieRatings = [ (1:1:nMovies)', zeros(nMovies,1)] ;

for j = 1:nMovies
    trainRatings = trainData(trainData(:,2)==j,3);
    testRatings = testData(testData(:,2)==j,3);
    movieRatings(j,2) = mean(trainRatings);
    
    if ~isempty(trainRatings)
        trainMseM = trainMseM + sum((trainRatings-movieRatings(j,2)).^2);
        trainNum = trainNum + length(trainRatings);
        if ~isempty(testRatings)
            testMseM = testMseM + sum((testRatings-movieRatings(j,2)).^2);
            testNum = testNum + length(testRatings);
        end 
    end
end

trainMseM = trainMseM/trainNum;
testMseM = testMseM/testNum;

% Print Results to Console
fprintf('Movie Rating Prediction (User-Independent)\n');
fprintf('Training Error: %3.6f\nTest Error: %3.6f\n',trainMseM, testMseM);

