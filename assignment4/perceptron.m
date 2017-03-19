function [wOpt] = perceptron(X,y, maxIter)

% set dimensions
[d,N] = size(X);

% initialise weights
wOpt = zeros(d+1,1);

% append the unit values at the begining of every vector x
X = [ones(1,N); X];

trainErrs = 2*ones(maxIter,1); 
weights = zeros(d+1,maxIter);
numUpdates = 0;
errorsExist = true;  
reachedMaxIter = false;

% Iterate until no mis-classified points left OR max iterations reached
while errorsExist && ~reachedMaxIter
    
    errorsExist = false;  
    for j = 1:N
        if sign(y(j) * (wOpt' * X(:,j))) ~= 1
            wOpt = wOpt + y(j)*X(:,j);
            numUpdates = numUpdates + 1;
            errorsExist = true;
            
            % Calculate training error at this update
            errChecks = sign(y' .* (wOpt' * X));
        
            trainErrs(numUpdates) = sum(errChecks ~= 1)/N;
        
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

[~,minErrIndex] = min(trainErrs);
wOpt = weights(:,minErrIndex);

end