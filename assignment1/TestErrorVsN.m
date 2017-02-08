% Question 3(b) Part 2 
% Running the perceptron algorithm (once) for N = [100, 200, 300, 400, 500]
% We generate the Test Error vs Number of Training Points graph

% Notation
% N: Number of (training) feature vectors
% M: Number of (test) feature vectors
% X: 2xN matrix containing N 2-dimensional feature vectors
% y: Nx1 binary classification values 

% Perceptron      -> x2 = a*x1 + b;
% True Classifier -> x2 = 1*x1 + c;

clear
close all

% Initialise parameters
c = 0.1;
M = 1e5;
testErrProbs = zeros(5,1);

% Obtain error calculations for [100, 200, 300, 400, 500] training points
for k = 1:1:5
    rng('default')
    N = k*100;
    
    % Initialise (x,y) pairs
    X = zeros(2,N);
    y = zeros(N,1);

    % Generate (x,y) pairs
    j=0;
    while j ~= N
        x = rand(2,1);
        if( x(2) - x(1) - c < 0) 
            j = j + 1;
            X(:,j) = x; y(j) = -1;
        else
            j = j + 1;
            X(:,j) = x; y(j) = 1;
        end 
    end
    
    % Find optimal weights
    [wOpt, iter] = perceptron(X,y);
    b = -wOpt(1)/wOpt(3);
    a = -wOpt(2)/wOpt(3);
    
    % Calculate testing error
    testErrProbs(k) = calcError(M,wOpt,c);
    
    % Print results to console
    fprintf('N= %d; iter= %d; err= %3.3f%%\n' , N, iter, 100*testErrProbs(k));
end

% Plot error values
t = 100:100:500;
plot(t, testErrProbs, 'o-' ,'Linewidth', 5, 'Markersize', 20);

% Figure options
title('Empirical Test Error vs Number of Training Points','FontSize',46);
xlabel('Number of Training Points','FontSize',36);
ylabel('Error Probability','FontSize',36);
grid on
grid minor
set(gca,'fontsize',32);
