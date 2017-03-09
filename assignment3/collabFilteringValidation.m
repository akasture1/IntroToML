% Author: Anand Kasture (ak7213)
% March 2017

% EE3-23 Machine Learning Assignment 3
% Q3D - Collaborative Filtering
clc
clear
close all

%% Stochastic Gradient Descent (SGD) Setup
gamma = 0.1;

% Max Epochs
epoch = 1;
maxEpochs = 50;

% Load Train/Test Data (userId, movieId, rating)
trainData = csvread('./movie-data/ratings-train.csv',1);
testData = csvread('./movie-data/ratings-test.csv',1);

% Load Features (movieId, feat1, ..., feat18)
features  = csvread('./movie-data/movie-features.csv',1);
[numMovies, ~] = size(features);

% Useful SGD Functions
getError = @(r, u, v) r - (u*v');
nextUserFeat = @(r, u, v, gamma, lambda) u + gamma*(getError(r,u,v)*v - lambda*u);
nextMovieFeat = @(r, u, v, gamma, lambda) v + gamma*(getError(r,u,v)*u - lambda*v);

%% Validation Setup
% Model Set
lambdas = 0.9:0.1:1.4;
Ks = 10:1:15;

% Validation Error Vector (per model)
valErrors = zeros(length(Ks),length(lambdas));

% K-Fold Cross Validation Indices
indices = crossvalind('Kfold', length(trainData), 10);


%% Validate best {lambda,K} pair
fprintf('=-=-=-=-=-=-=-=---Starting Validation---=-=-=-=-=-=-=-=\n\n')
% For each {lambda,K} pair
for i = 1:1:length(lambdas)
    lambda = lambdas(i);
    for k = 1:1:length(Ks)
        K = Ks(k);
        fprintf('---------------[lambda=%f] [K=%d]---------------',lambda,K);
        % For each Train-Validiation Data Configuration
        for valIndex = 1:1:max(indices)
            
            valDataSegment = trainData(indices==valIndex,:);        
            trainDataSegment = trainData(indices~=valIndex,:);
            
            [numValEntries,~] = size(valDataSegment);
            [numTrainEntries,~] = size(trainDataSegment);
 
            valUserIds = unique(valDataSegment(:,1));
            trainUserIds = unique(trainDataSegment(:,1));
            
            % Initialise RANDOM feature vectors
            userFeatures = rand(length(trainUserIds), K);
            movieFeatures = rand(numMovies, K);
            
            epoch = 1;
            bestValError = realmax;
            fprintf('\n[valIndex=%d]; epoch:',valIndex);
            while epoch <= maxEpochs
                fprintf('%d',epoch);
                for j = 1:1:numTrainEntries
                    trainEntry = trainDataSegment(j,:);
                    r = trainEntry(3);
                    u = userFeatures(trainEntry(1),:);
                    v = movieFeatures(trainEntry(2),:);

                    userFeatures(trainEntry(1),:) = nextUserFeat(r, u, v, gamma, lambda);
                    movieFeatures(trainEntry(2),:) = nextMovieFeat(r, u, v, gamma, lambda);
                end    
        
                valError = 0;
                for user = 1:1:length(trainUserIds)
                    currTrainData = trainDataSegment(trainDataSegment(:,1)==user,:);
                    currValData = valDataSegment(valDataSegment(:,1)==user,:);
                    U = userFeatures(user,:);

                    % Calculate Validation Error
                    R = currValData(:,3);
                    V = movieFeatures(currValData(:,2),:); 
                    valError = valError + sum((R - V*U').^2);
                end
                valError = (1/numValEntries)*valError;
                if valError < bestValError
                    bestValError = valError;
                end
                epoch = epoch + 1;
            end % epochs
            valErrors(k,i) = valErrors(k,i) + bestValError;
        end % train-validation config
        valErrors(k,i) = (1/max(indices)) * valErrors(k,i);
        fprintf('\n---------------[lambda=%f] [K=%d] Validation Error=%f\n\n',lambda,K,valErrors(k,i));
    end %K
end %lambda

%% DONE!



