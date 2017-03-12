function [minIndex, lambdas, valErrors, trainErrors] = getLambdaAllUsers(trainData, features, K)
% Using validation to generate ideal value for lambda, for all user

% Training Data Size
[N,~] = size(trainData);

% Model Set
lambdas = linspace(0.01, 2, 100);

% K-Fold Cross Validation Indices
indices = crossvalind('Kfold', N, K);

% One index for each model
valErrors = zeros(length(lambdas),1);
trainErrors = zeros(length(lambdas),1);

for i = 1:1:length(lambdas)
    valNum = 0;
    trainNum = 0;
    lambda = lambdas(i);
    for valIndex = 1:1:max(indices)
        valDataSegment = trainData(indices==valIndex,:);        
        trainDataSegment = trainData(indices~=valIndex,:);
        
        valUserIds = unique(valDataSegment(:,1));
        trainUserIds = unique(trainDataSegment(:,1));

        for j = 1:1:length(trainUserIds)
            userId = trainUserIds(j);
            currTrainData = trainDataSegment(trainDataSegment(:,1)==userId,:);
            
            y = currTrainData(:,3);
            Z = features(currTrainData(:,2),:); 
            Z(:,1) = 1;
            wReg = ridgeRegression(Z,y,lambda);
            trainErrors(i) = trainErrors(i) + sum((y - Z*wReg).^2);
            trainNum = trainNum + length(y);
            
            if ~isempty(valUserIds(valUserIds(:,1)==userId))
                currValData = valDataSegment(valDataSegment(:,1)==userId,:);
                y = currValData(:,3);
                Z = features(currValData(:,2),:); 
                Z(:,1) = 1;
                valErrors(i) = valErrors(i) + sum((y - Z*wReg).^2);
                valNum = valNum + length(y);
            end
        end
    end
    valErrors(i) = (1/valNum).*valErrors(i);
    trainErrors(i) = (1/trainNum).*trainErrors(i);
    fprintf('Lambda:%f valError:%f\n',lambda,valErrors(i));
end

[~,minIndex] = min(valErrors);
%lambda = lambdas(minIndex);
end