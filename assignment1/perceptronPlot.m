% Question 3(a)
% This script runs the perceptron learning algorithm on random training data
% We plot the resulting classifications on the x1,x2 plane
% The number of iterations and the test error is printed to the console

% Notation
% N: Number of (training) feature vectors
% M: Number of (test) feature vectors
% X: 2xN matrix containing N 2-dimensional feature vectors
% y: Nx1 binary classification values 

% Perceptron      -> x2 = a*x1 + b;
% True Classifier -> x2 = 1*x1 + c;

clear
close all
rng('default')

% Initialise parameters
N = 100;
y = zeros(N,1);
c = 0.1;
M = 1e5;

% Generate Training Data
X = rand(2,N);
logical = (X(2,:) - X(1,:) - c > 0)';
y(logical==true) = 1;  
y(logical==false) = -1;   

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

% Empirically calculate the test error
testErrProb = calcError(M,wOpt,c);

% Print results to console
fprintf('N= %d; iter= %d; err= %3.3f%%\n' , N, iter, 100*testErrProb);

% Plot line defined by the vector w
t = 0:0.01:1;
trueLine = t + c;
perceptronLine = a*t + b;

p1 = plot(t, trueLine, '--k' ,'Linewidth', 3);                              % true separator
p2 = plot(t, a*t + b, 'Color', [ 0.4660 0.6740 0.1880] ,'Linewidth', 5);    % perceptron separator

% Figure options
title('The Perceptron Learning Algorithm; N=100','FontSize',46);
xlabel('x_{1}','FontSize',36);
ylabel('x_{2}','FontSize',36);
legend([p1 p2],'true separator','perceptron separator','Location','southeast');
grid on
grid minor
set(gca,'fontsize',32);
xlim([0 1]);ylim([0 1])
h = fill([t, fliplr(t)], [perceptronLine, fliplr(trueLine)], 'y');
set(h,'facealpha',.5)
%patch([t fliplr(t)], [perceptronLine,fliplr(trueLine)],'y')
