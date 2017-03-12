% Author: Anand Kasture (ak7213)
% March 2017

% EE3-23 Machine Learning Assignment 3
% Q3D - Collaborative Filtering
clc
clear
close all

%% Stochastic Gradient Descent (SGD) Setup
gamma = 0.01;

% Max Epochs
epoch = 1;
maxEpochs = 100;

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
lambdas = [0.0001 0.001 0.01 0.1 1];
Ks = [1 5 10 15];

% Validation Error Vector (per model)
valErrors3D = zeros(length(Ks),length(lambdas), maxEpochs);

% K-Fold Cross Validation Indices
indices = crossvalind('Kfold', length(trainData), 10);


%% Validate best {lambda,K} pair
fprintf('=-=-=-=-=-=-=-=---Starting Validation---=-=-=-=-=-=-=-=\n')
% For each {lambda,K} pair
for i = 1:1:length(lambdas)
    lambda = lambdas(i);
    for k = 1:1:length(Ks)
        K = Ks(k);
        fprintf('\n---------------[lambda=%f] [K=%d]---------------',lambda,K);
        % For each Train-Validiation Data Configuration
        bestValErrors = realmax*ones(maxEpochs,1);
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
            valErrors = zeros(maxEpochs,1);
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
        
                for user = 1:1:length(trainUserIds)
                    currTrainData = trainDataSegment(trainDataSegment(:,1)==user,:);
                    currValData = valDataSegment(valDataSegment(:,1)==user,:);
                    U = userFeatures(user,:);

                    % Calculate Validation Error
                    R = currValData(:,3);
                    V = movieFeatures(currValData(:,2),:); 
                    valErrors(epoch) = valErrors(epoch) + sum((R - V*U').^2);
                end
                valErrors(epoch) = (1/numValEntries)*valErrors(epoch);
                if valErrors(epoch) < bestValErrors(epoch) 
                    bestValErrors(epoch) = valErrors(epoch);
                end
                epoch = epoch + 1;
            end % epochs
        end % train-validation config
        valErrors3D(k,i,:) = bestValErrors;
    end %K
end %lambda

figure
M = squeeze(valErrors3D(1,:,:))'; p1 = plot(M(:,1:end-1), 'r', 'LineWidth', 2); %legend('K = 1');
hold on
M = squeeze(valErrors3D(2,:,:))'; p2 = plot(M(:,1:end-1), 'g', 'LineWidth', 2); %legend('K = 5');
M = squeeze(valErrors3D(3,:,:))'; p3 = plot(M(:,1:end-1), 'b', 'LineWidth', 2); %legend('K = 10');
M = squeeze(valErrors3D(4,:,:))'; p4 = plot(M(:,1:end-1), 'y', 'LineWidth', 2); %legend('K = 15');
legend([p1(1) p2(1) p3(1) p4(1)], {'K=1','K=5','K=10','K=15'});
title('Cross-Validation Error for each {Lambda, K} Configuration','FontSize',46);
xlabel('Epochs','FontSize',36);
ylabel('Error','FontSize',36);
grid on
grid minor
set(gca,'fontsize',32);

%% DONE!