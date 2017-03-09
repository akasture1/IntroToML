function [lambda] = getLambdaPerUser(currTrainData, Z, K)
% Using validation to generate ideal value for lambda, for a single user

% Training Data Size
[N,~] = size(currTrainData);

% Model Set
lambdas = 0.1:0.1:5;

% K-Fold Cross Validation Indices
indices = crossvalind('Kfold', N, K);

% One index for each model
valErrors = zeros(length(lambdas),1);

% Iterate over every training-validation data set config
for valIndex = 1:1:max(indices)
    valData = currTrainData(indices==valIndex,:);
    valRatings = valData(:,3);
    valFeatures = Z(indices==valIndex,:);
    
    trainData = currTrainData(indices~=valIndex,:);
    trainRatings = trainData(:,3);
    trainFeatures = Z(indices~=valIndex,:);
    
    % Compute cross validation error for each model
    for j = 1:1:length(lambdas)
        lambda = lambdas(j);
        wReg = ridgeRegression(trainFeatures,trainRatings,lambda);
        valErrors(j) = valErrors(j) + sum((valRatings - valFeatures*wReg).^2);
    end
end

valErrors = (1/N).*valErrors;
[~,j] = min(valErrors);
lambda = lambdas(j);
end

