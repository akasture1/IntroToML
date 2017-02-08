% Question 3(c) Part 1
% Using the perceptronPro algorithm to classify the digits 2 and 8

% --------Raw Data
% Filter for digits 2 and 8
clear all
close all

X = importdata('./data/zip.train');
X = X(X(:,1)==2 | X(:,1)==8 ,:);
y = X(:,1);

% Map digit -> class : {2,8} -> {1,-1}
X = X(:,2:end)';
y(y(:)==2) = 1;
y(y(:)==8) = -1;

[d1,N] = size(X);

maxIter = 1e5;
%[~,iter] = perceptronPlus(X, y, maxIter);
[w1,iter1] = perceptron(X,y);
[w2,iter2] = perceptronPlus(X,y,maxIter);
%[w3,iter3] = perceptronPro(X,y,maxIter,1);

fprintf('perceptron carried out %d updates on %d [%d x 1] training data (raw greyscale)\n', iter1, N, d1);
fprintf('perceptronPlus carried out %d updates on %d [%d x 1] training data (raw greyscale)\n', iter2, N, d1);
%fprintf('perceptronPro carried out %d updates on %d [%d x 1] training data (raw greyscale)\n', iter3, N, d1);


% --------Feature Vectors
% Note:: This 2-dimensional data is not linearly separable
% We will apply a modified perceptron algorithm

% {X,y} -> Training Data
% {R,s} -> Test Data

X = importdata('./data/features.train');
X = X(X(:,1)==2 | X(:,1)==8 ,:);
y = X(:,1);

% Map digit -> class : {2,8} -> {1,-1}
X = X(:,2:end)';
y(y==2) = 1;
y(y==8) = -1;

[d1,N] = size(X);

% Linear Regressor
x1 = X(1,:)';   % first component of every feature vector
x2 = X(2,:)';   % second component of every feature vector

XBar=[ones(N,1) x1];
wLR = pinv(XBar)*x2;

% Run perceptronPro on feature vector training data
maxIter = 1e5;
[w2, iter2, errorProbs] = perceptronPlus(X, y, maxIter);
[w3,iter3] = perceptronPro(X,y,maxIter,wLR,0);

b2 = -w2(1)/w2(3);
a2 = -w2(2)/w2(3);

b3 = -w3(1)/w3(3);
a3 = -w3(2)/w3(3);

fprintf('perceptronPlus carried out %d updates on %d [%d x 1] training data (feature vectors)\n', iter2, N, d1);
fprintf('perceptronPro carried out %d updates on %d [%d x 1] training data (feature vectors)\n', iter3, N, d1);

% Plot Test Data with the Perceptron Line
R = importdata('data/features.test');
R = R(R(:,1)==2 | R(:,1)==8 ,:);
s = R(:,1);

% Map digit -> class : {2,8} -> {1,-1}
R = R(:,2:end)';
s(s(:)==2) = 1;
s(s(:)==8) = -1;

[d2,M] = size(R);

fprintf('Plotting %d [%d x 1] input test data (feature vectors) with the perceptron classifier\n', M, d2);

figure(1)
class2 = find(s== 1);
class8 = find(s==-1);

plot(R(1,class2), R(2,class2), 'x', 'MarkerSize', 15, 'LineWidth', 3);
hold on
plot(R(1,class8), R(2,class8), 'o', 'MarkerSize', 15, 'LineWidth', 3);

t = 0:0.01:1;
perceptronPlusLine = a2*t + b2;
perceptronProLine = a3*t + b3;
linearRegressorLine = wLR(2)*t + wLR(1);

plot(t, perceptronPlusLine, 'k' ,'Linewidth', 3);
plot(t, perceptronProLine, 'g' ,'Linewidth', 3);
plot(t, linearRegressorLine, 'y' ,'Linewidth', 3);

% Figure Options
title('Handwritten Digit Classification with the Perceptron Learning Algorithm','FontSize',46);
xlabel('x_{1} (Intensity)','FontSize',36);
ylabel('x_{2} (Symmetry)','FontSize',36);
legend('Digit 2','Digit 8','Perceptron', 'PerceptronPro', 'Linear Regressor');
grid on
grid minor
set(gca,'fontsize',32);
%axis([floor(min(X(1,:))) ceil(max(X(1,:))) floor(min(X(2,:))) ceil(max(X(2,:)))])
axis([0.05 0.55 -8 0])
hold off

