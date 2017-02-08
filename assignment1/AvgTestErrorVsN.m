% Question 3(b) Part 3 
% Running T = 100 perceptron trials for N = [100, 200, 300, 400, 500]
% We generate the 90% Confidence Interval Test Error vs Number of Training Points graph

% Notation
% N: Number of (training) feature vectors
% M: Number of (test) feature vectors
% T: Number of trials
% X: 2xN matrix containing N 2-dimensional feature vectors
% y: Nx1 binary classification values 

% Perceptron      -> x2 = a*x1 + b;
% True Classifier -> x2 = 1*x1 + c;
clear
close all
rng('default')

% Initialise parameters
c = 0.1;
M = 1e5;
T = 100;
testErrProbs = zeros(100,5);

% Obtain error calculations for [100, 200, 300, 400, 500] training points
% Carry out T trials for each N value
for k = 1:1:5
    
    N = k*100;
    
    % Initialise (x,y) pairs
    X = zeros(2,N);
    y = zeros(N,1);
    
    % Iterate through each trial
    for j = 1:T
        % Generate (x,y) pairs
        i=0;
        while i ~= N
            x = rand(2,1);
            if( x(2) - x(1) - c < 0) 
                i = i + 1;
                X(:,i) = x; y(i) = -1;
            else
                i = i + 1;
                X(:,i) = x; y(i) = 1;
            end 
        end
        
        % Find optimal weights
        [wOpt, iter] = perceptron(X,y);
        b = -wOpt(1)/wOpt(3);
        a = -wOpt(2)/wOpt(3);

        % Calculate testing error
        testErrProbs(j,k) = calcError(M,wOpt,c);
        fprintf('N= %d; Trial= %d; iter= %d; err= %3.3f%%\n', N, j, iter, 100*testErrProbs(j,k));
    end
end

% Process data from errors matrix
topLine = zeros(5,1);
bottomLine = zeros(5,1);
means = zeros(5,1);
for j = 1:5
    topLine(j) = prctile(testErrProbs(:,j),95);
    bottomLine(j) = prctile(testErrProbs(:,j),5);
    means(j) = mean(testErrProbs(:,j));
end

% Plot 90% confidence interval test error, and mean errors
figure(1)
t = 100:100:500;
plot(t, topLine,    'ko-' ,'Linewidth', 4, 'Markersize', 20);
hold on
plot(t, bottomLine, 'ko-' ,'Linewidth', 4, 'Markersize', 20);
plot(t, means,      'o-' ,'Linewidth', 5, 'Markersize', 20);

h = fill([t, fliplr(t)], [topLine', fliplr(bottomLine')], [  0    0.4470    0.7410]);
set(h,'facealpha',.3)

% Figure options
title('90% Confidence Interval','FontSize',46);
xlabel('Number of Training Points','FontSize',36);
ylabel('Error Probability','FontSize',36);
legend('95^{th} Percentile','5^{th}   Percentile','Mean');
grid on
grid minor
set(gca,'fontsize',32);
