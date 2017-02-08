function [ maxIters ] = getMaxUpdates(X,y,w)
   
    [~,n] = size(X);
    
    % append the unit values at the begining of every vector x
    X = [ones(1,n); X];
       
    % Compute rho
    rho = min((y') .* (w'*X));
    
    % Compute R
    R = max( sqrt(sum(abs(X').^2,2)) );
    
    % Return theoretical maximum number of training errors
    maxIters = (R^2) * (norm(w)^2) / (rho^2);
    maxIters = ceil(maxIters);
end

