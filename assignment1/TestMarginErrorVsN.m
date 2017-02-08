% Question 3(b) Part 4.1
% Running the perceptron algorithm (once) for N = [100, 200, 300, 400, 500]
% We generate the Test Error vs Number of Training Points graph
% Investigation of a margin defined by gamma

% Notation
% N: Number of (training) feature vectors
% M: Number of (test) feature vectors
% X: 2xN matrix containing N 2-dimensional feature vectors
% y: Nx1 binary classification values 

% Perceptron      -> x2 = a*x1 + b;
% True Classifier -> x2 = 1*x1 + c;
% Repeat error calculation for [100, 200, 300, 400, 500] training points

clear
close all

% Initialise parameters
c = 0.1;
gamma = 0.001;
M = 1e5;
testErrProbs = zeros(5,1);
updates = zeros(5,1);
maxUpdates = zeros(5,1);

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
        if(x(2) - x(1) - c > gamma)
            j = j + 1;
            X(:,j) = x; y(j) = 1;
          
        elseif(x(2) - x(1) - c < -gamma)
            j = j + 1;
            X(:,j) = x; y(j) = -1;
        end
    end
       
    % Find optimal weights
    [wOpt, updates(k)] = perceptron(X,y);
    b = -wOpt(1)/wOpt(3);
    a = -wOpt(2)/wOpt(3);
    
    % Find theoretical max. number of iterations
    maxUpdates(k) = getMaxUpdates(X,y,wOpt);
    fprintf('maxUpdates= %d\n' ,maxUpdates(k));

    % Calculate testing error
    testErrProbs(k) = calcMarginError(M,wOpt,c,gamma);
    
    % Print results to console
    fprintf('N= %d; iter= %d; gamma= %1.4f; err= %3.4f%%\n\n' , N, updates(k), gamma, 100*testErrProbs(k));
end

% Plot error values
figure(1)
t = 100:100:500;
plot(t, testErrProbs, 'o-' ,'Linewidth', 5, 'Markersize', 20);

% Figure options
title(['Empirical Test Error vs Number of Training Points;    \gamma =' num2str(gamma) ''],'FontSize',46);
xlabel('Number of Training Points','FontSize',36);
ylabel('Error Probability','FontSize',36);
grid on
grid minor
set(gca,'fontsize',32);

figure(2)
semilogy(t, updates, 'o-' ,'Linewidth', 5, 'Markersize', 20);
hold on
semilogy(t, maxUpdates, 'o-' ,'Linewidth', 5, 'Markersize', 20);

% Figure options
title(['Number of Perceptron Updates vs Number of Training Points;   \gamma =' num2str(gamma) ''],'FontSize',46);
xlabel('Number of Training Points','FontSize',36);
ylabel('Number of Updates','FontSize',36);
legend('Iterations','Theoretical Max Iterations');
grid on
grid minor
set(gca,'fontsize',32);