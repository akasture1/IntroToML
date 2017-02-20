function [wOpt, minError] = perceptronPlus(X,y,maxIter)

% set dimensions
[d,n] = size(X);

% initialise weights
wOpt = zeros(d,1);
% wOpt = [0;wLR];

%xNorms = sqrt(sum(abs(X').^2,2));
%[xNormsDesc,xNormsDescIndices] = sort(xNorms, 'descend'); 

errors = 2*ones(maxIter,1); 
weights = zeros(d,maxIter);
nCorrections = 0;
errorsExist = true;  
reachedMaxIter = false;

% Iterate until no mis-classified points left OR max iterations reached
while errorsExist && ~reachedMaxIter
    
    errorsExist = false;  
    for j = 1:n
        if sign(wOpt' * X(:,j)) ~= y (j)
            wOpt = wOpt + y(j)*X(:,j);
            nCorrections = nCorrections + 1;
            errorsExist = true;
            
            % Calculate training error at this update
            percepClass = y' .* (wOpt' * X);
            errors(nCorrections) = sum(percepClass < 0)/n;
        
%           percepClass = sign(wOpt' * X)';
%           percepClass(percepClass==0) = 1;
%           errorProbs(numUpdates) = numel(find(percepClass~=y))/N;

            % Keep track of the weight vector at each iteration
            weights(:,nCorrections)= wOpt;
            
            % Check if we have reached maxIter
            if(nCorrections == maxIter)
                reachedMaxIter = true;
                break;
            end
            
        end
    end
end

[minError,minErrIndex] = min(errors);
wOpt = weights(:,minErrIndex);

end