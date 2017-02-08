function [wOpt, numUpdates] = perceptron(X,y)
% Simple MATLAB implementation of the Perceptron Learning Algorithm

% set dimensions
[d,N] = size(X);

% initialise weights
wOpt = zeros(d+1,1);

% append the unit values at the begining of every vector x
X = [ones(1,N); X];

% iterate until no mis-classified points left
errorsExist = true;         
numUpdates = 0;
tol = 1e-4;

while errorsExist
    foundError = false;
    for j = 1:N
        if(y(j) * (wOpt' * X(:,j)) < tol)
            wOpt = wOpt + y(j)*X(:,j);
            numUpdates = numUpdates + 1;
            foundError = true;
        end
    end

    % check if finished
    if foundError == false
        break;
    end
    
end
end