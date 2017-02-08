function [wOpt, numUpdates, errorProbs, weights, minErrIndex] = perceptronPlus(X,y, maxIter)
% Improvements
% 1. Outputs training error probabilities and weights at each iteration
% 2. Will stop after maxIter iterations

% set dimensions
[d,N] = size(X);

% initialise weights
wOpt = zeros(d+1,1);
% wOpt = [0;wLR];

% append the unit values at the begining of every vector x
X = [ones(1,N); X];

%xNorms = sqrt(sum(abs(X').^2,2));
%[xNormsDesc,xNormsDescIndices] = sort(xNorms, 'descend'); 

tol = 1e-4;
errorProbs = 2*ones(maxIter,1); 
weights = zeros(d+1,maxIter);
numUpdates = 0;
errorsExist = true;  
reachedMaxIter = false;

% Iterate until no mis-classified points left OR max iterations reached
while errorsExist && ~reachedMaxIter
    
    errorsExist = false;  
    for j = 1:N
        if y(j) * (wOpt' * X(:,j)) < tol
            wOpt = wOpt + y(j)*X(:,j);
            numUpdates = numUpdates + 1;
            errorsExist = true;
            
            % Calculate training error at this update
            percepClass = y' .* (wOpt' * X);
            errorProbs(numUpdates) = sum(percepClass < 0)/N;
        
%           percepClass = sign(wOpt' * X)';
%           percepClass(percepClass==0) = 1;
%           errorProbs(numUpdates) = numel(find(percepClass~=y))/N;

            % Keep track of the weight vector at each iteration
            weights(:,numUpdates)= wOpt;
            
            % Check if we have reached maxIter
            if(numUpdates == maxIter)
                reachedMaxIter = true;
                break;
            end
            
        end
    end
end


[~,minErrIndex] = min(errorProbs);
wOpt = weights(:,minErrIndex);
weights = weights(:,1:numUpdates);
errorProbs = errorProbs(1:numUpdates);

end