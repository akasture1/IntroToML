clear
close all
rng('default')

nTestData = 1e4;
maxPercepIter = 1e4;

trainData=[10,100,10000];
qMax=5;
nTrials=1e3;

trainErrors = zeros(length(trainData),qMax,nTrials);
erm = zeros(length(trainData),qMax,nTrials);
srm = zeros(length(trainData),qMax,nTrials);

idealFunc = @(x1,x2)[x1.^3 x1.^2 x1 x2];
idealWeights = [-1 3 -2 1];

growthFunc = @(n,vcDim)(2*n*exp(1)/vcDim)^vcDim;
omega = @(n,vcDim,del) sqrt(8*log(4*growthFunc(n,vcDim)./del)./n);

del = 0.05; % 95% Confidence Interval
const = 1;  % Hueristic modification to the Complexity Term

for k = 1:length(trainData)
    
    nTrainData = trainData(k);
    
    for j = 1:nTrials
        
        x1Train = 2.5*rand(nTrainData,1);
        x2Train = 3*rand(nTrainData,1)-1;
        
        flipMult = randsample([1,-1],nTrainData,true,[0.9, 0.1])';
        yTrain = (sign(idealFunc(x1Train,x2Train) * idealWeights').*flipMult);

        for i = 1:qMax
            fprintf('nTrainData: %d, Trial: %d, Q: %d\n',nTrainData,j,i)
            
            q = i-1;
            ZTrain = polyTransform([x1Train x2Train],q);           
            
            % Training Error
            [wOpt, trainErrors(k,i,j)] = perceptronPlus(ZTrain,yTrain,maxPercepIter);

            % Test Error (erm and srm)
            x1Test = 2.5*rand(nTestData,1);
            x2Test = 3*rand(nTestData,1)-1;

            ZTest = polyTransform([x1Test x2Test],q);

            yTest = sign(wOpt'*ZTest)';
            yTestIdeal = sign(idealFunc(x1Test,x2Test) * idealWeights');

            % determine erm and srm risk
            erm(k,i,j) = numel(find(yTest~=yTestIdeal))/nTestData;
            srm(k,i,j) = erm(k,i,j) + const*omega(nTrainData, q+1, del);
        end
    end
end