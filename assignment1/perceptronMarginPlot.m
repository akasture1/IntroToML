% Question 3(a)
% This script runs the perceptron learning algorithm on random training data
% We plot the resulting classifications on the x1,x2 plane
% The number of iterations and the test error is printed to the console
% Investigating the impact of a margin

% Notation
% N: Number of (training) feature vectors
% M: Number of (test) feature vectors
% X: 2xN matrix containing N 2-dimensional feature vectors
% y: Nx1 binary classification values 

% Perceptron       -> x2 = a*x1 + b;
% True Classifier1 -> x2 = 1*x1 + c + gamma;
% True Classifier2 -> x2 = 1*x1 + c - gamma;

clear
close all
rng('default')

% Initialise parameters
N = 100;
X = zeros(2,N);
y = zeros(N,1);
c = 0.1;
gamma = 0.001;
M = 1e5;

j=0;
while j ~= N
    % Generate Training Data
    x = rand(2,1);
    if(x(2) - x(1) - c > gamma)
        j = j + 1;
        X(:,j) = x;
        y(j) = 1;

    elseif(x(2) - x(1) - c < -gamma)
        j = j + 1;
        X(:,j) = x;
        y(j) = -1;
    end
end

% Plot Training Data
figure(1)
classAbove = find(y== 1);
classBelow = find(y==-1);
plot(X(1,classAbove), X(2,classAbove), 'x', 'MarkerSize', 20, 'LineWidth', 5);
hold on
plot(X(1,classBelow), X(2,classBelow), 'o', 'MarkerSize', 20, 'LineWidth', 5);

% Find optimal weights
[wOpt, iter] = perceptron(X,y);
b = -wOpt(1)/wOpt(3);
a = -wOpt(2)/wOpt(3);

% Empirically calculate the testing error
testErrProb = calcMarginError(M,wOpt,c,gamma);

% Print results to console
fprintf('N= %d; iter= %d; gamma= %1.4f; err= %3.4f%%\n' , N, iter, gamma, 100*testErrProb);

% Plot line defined by the vector w
t = 0:0.01:1;

p1 = plot(t, t + c + gamma, 'k','Linewidth', 3);                            % true separator
p2 = plot(t, t + c - gamma, 'k' ,'Linewidth', 3);                           % true separator
p3 = plot(t, a*t + b, 'Color', [ 0.4660 0.6740 0.1880] ,'Linewidth', 5);    % perceptron separator

% Figure options
title(['The Perceptron Learning Algorithm;    \gamma =' num2str(gamma) ''],'FontSize',46);
xlabel('x_{1}','FontSize',36);
ylabel('x_{2}','FontSize',36);
legend([p1 p3],'true separators','perceptron separator','Location','southeast');
grid on
grid minor
set(gca,'fontsize',32);
xlim([0 1]);ylim([0 1])
