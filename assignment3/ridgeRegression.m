function [ wReg ] = ridgeRegression( X, y, lambda )
    [~,n] = size(X);
    wReg = (X'*X + lambda*eye(n)) \ (X'*y);
end

