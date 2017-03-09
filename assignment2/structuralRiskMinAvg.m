% Assignment 2 - Problem 5b
% ERM and SRM Setup 

clear
close all
rng('default')

% Number of Test Points
nTestData = 1e4;

% Maximum Number of Perceptron Updates
maxPercepIter = 1e4;

% Number of Training Data Points
trainData=[10,100,10000];

% Iterate through q = [0,...,4]
qMax=5;

% Number of Trials/Repetitions
nTrials=1e2;

% 3-D Error Matrices
ermTrainErrors = zeros(length(trainData),qMax,nTrials);
ermTestErrors = zeros(length(trainData),qMax,nTrials);

srmTrainErrors = zeros(length(trainData),nTrials);
srmTestErrors = zeros(length(trainData),nTrials);

% Weight Matrix
ermWeights = zeros(sum(1:1:qMax)+qMax,length(trainData),nTrials);
srmWeight = [];

% Lambda Functions 
idealFunc = @(x1,x2)[x1.^3 x1.^2 x1 x2];
idealWeights = [-1 3 -2 1];

growthFunc = @(n,vcDim)(2*n*exp(1)/vcDim)^vcDim;
omega = @(n,vcDim,del) sqrt(8*log(4*growthFunc(n,vcDim)./del)./n);

% Complexity Term Parameters
del = 0.05;         % 95% Confidence Interval
del = 0.2 * del;    % 5 Equally Weighted Polynomial Hypothesis Sets
const = 0.01;          % Hueristic Modification 

for k = 1:length(trainData)
    nTrainData = trainData(k);    
    
    for j = 1:nTrials   
        
        % Set up Training Data
        x1Train = 2.5*rand(nTrainData,1);
        x2Train = 3*rand(nTrainData,1)-1;
        
        flipMult = randsample([1,-1],nTrainData,true,[0.9, 0.1])';
        yTrain = (sign(idealFunc(x1Train,x2Train) * idealWeights').*flipMult);
        
        minTrainingError = realmax;
        bestSetIndex = 0;
        for i = 1:qMax
            fprintf('nTrainData: %d, Trial: %d, Q: %d\n',nTrainData,j,i)
            q = i-1;
            
            % Non-Linear Transform
            ZTrain = polyTransform([x1Train x2Train],q);           
            
            % Run Perceptron and obtain best-case Training Error
            [w, ermTrainErrors(k,i,j)] = perceptronPlus(ZTrain,yTrain,maxPercepIter);
            
            ermWeights(sum(1:1:i):sum(1:1:i+1)-1,k,j) = w;
            
            if(ermTrainErrors(k,i,j) < minTrainingError)
                bestSetIndex = i;
                srmWeight = w;
                minTrainingError = ermTrainErrors(k,i,j);
                srmTrainErrors(k,j) = minTrainingError;
            end
            
            % Set up Test Data
            x1Test = 2.5*rand(nTestData,1);
            x2Test = 3*rand(nTestData,1)-1;

            ZTest = polyTransform([x1Test x2Test],q);

            yTestERM = sign(w'*ZTest)';
            yTestIdeal = sign(idealFunc(x1Test,x2Test) * idealWeights');

            % ERM Test Error
            ermTestErrors(k,i,j) = numel(find(yTestERM~=yTestIdeal))/nTestData;

            % SRM Test Error
            if(bestSetIndex == i)
                yTestSRM = sign(srmWeight'*ZTest)';
                srmTestErrors(k,j) = numel(find(yTestSRM~=yTestIdeal))/nTestData + const*omega(nTrainData, q+1, del);
            end
        end         
    end
end