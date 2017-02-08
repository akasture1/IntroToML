function [wOpt, numUpdates, errorProbs, weights, minErrIndex] = perceptronPro(X,y, maxIter, useMedian)
% Improvements
% 

% set dimensions
[d,N] = size(X);

% initialise weights
 wOpt = zeros(d+1,1);
%wOpt = [0;wLR];
 
% append the unit values at the begining of every vector x
X = [ones(1,N); X];

tol = 1e-4;
errorProbs = 2*ones(maxIter,1); 
weights = zeros(d+1,maxIter);
numUpdates = 0;
errorsExist = true;  
reachedMax = false;

%xNorms = sqrt(sum(abs(X').^2,2));
%[xNormsDesc,xNormsDescIndices] = sort(xNorms, 'descend'); 

% Iterate until no mis-classified points left OR max iterations reached

while (errorsExist && ~reachedMax)
    errorsExist = false;  
    
    rho = y' .* (wOpt' * X);
    %[rho, j] = min(rho);
    
    if(useMedian)   %use median
        medVal = median(rho(rho<0));
    if(isnan(medVal))
        [rho, j] = min(rho);
    else
        j = find(abs(rho-medVal)<0.05);
        if(isempty(j))
            [rho, j] = min(rho);
        elseif(length(j) > 1)
            [rho, j] = min(rho(j));
        else
            rho = rho(j);
        end
    end
    else            %use min
        [rho, j] = min(rho);
    end
    
    if (rho < tol)
        wOpt = wOpt + y(j)*X(:,j);
        numUpdates = numUpdates + 1;
        errorsExist = true;

        % Calculate training error at this update
        percepClass = y' .* (wOpt' * X);
        errorProbs(numUpdates) = sum(percepClass < 0)/N;
        
%         percepClass = sign(wOpt' * X)';
%         percepClass(percepClass==0) = 1;
%         errorProbs(numUpdates) = numel(find(percepClass~=y))/N;

        % Keep track of the weight vector at each iteration
        weights(:,numUpdates)= wOpt;

        % Check if we have reached maxIter
        if(numUpdates == maxIter)
            reachedMax = true;
            break;
        end
    end
end

[~,minErrIndex] = min(errorProbs);
wOpt = weights(:,minErrIndex);
weights = weights(:,1:numUpdates);
errorProbs = errorProbs(1:numUpdates);

end