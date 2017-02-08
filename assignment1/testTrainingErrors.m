clear all
close all

% Question 3(c) Part 2
% Comparing Training & Test Error for perceptronPlus

% --------Raw Data
% {X,y} -> Training Data
% {R,s} -> Test Data

X = importdata('./data/features.train');
%X = importdata('./data/zip.train');
X = X(X(:,1)==2 | X(:,1)==8 ,:);
y = X(:,1);

% Map digit -> class : {2,8} -> {1,-1}
X = X(:,2:end)';
y(y==2) = 1;
y(y==8) = -1;

[d1,N] = size(X);


% Obtain Training Error
maxIter = 1e5;
[wOpt, iter, errorProbs, weights, minErrIndex] = perceptronPro(X, y, maxIter,1);
%[wOpt, iter, errorProbs, weights, minErrIndex] = perceptronPlus(X, y, maxIter);

fprintf('perceptron carried out %d updates on %d [%d x 1] raw training data\n', iter, N, d1);

trainingError = errorProbs(minErrIndex);
fprintf('Training Error: %3.6f\n', trainingError);

% Obtain Test Error
R = importdata('./data/features.test');
%R = importdata('./data/zip.test');
R = R(R(:,1)==2 | R(:,1)==8 ,:);
s = R(:,1);

% Map digit -> class : {2,8} -> {1,-1}
R = R(:,2:end)';
s(s(:)==2) = 1;
s(s(:)==8) = -1;

[d2,M] = size(R);
percepClass = s' .* (wOpt' * [ones(1,M);R]);
testError = sum(percepClass < 0)/M;
fprintf('Test Error: %3.6f\n', testError);
