
% Question 3(c) Part 2
% Training & Test Error vs Iterations for perceptron & perceptronPlus

% --------Raw Data
% {X,y} -> Training Data
% {R,s} -> Test Data
clear all
close all

%X = importdata('HandwritingData/zip.train');
X = importdata('/data/features.train');
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

% Obtain Training Error
maxIter = 1e5;
%[wOpt, iter, trainingError, weights, minErrIndex] = perceptronPro(X, y, maxIter, wLR, 1);
[wOpt, iter, trainingError, weights, minErrIndex] = perceptronPlus(X, y, maxIter, wLR);

% Obtain Test Error
%R = importdata('HandwritingData/zip.test');
R = importdata('data/features.test');
R = R(R(:,1)==2 | R(:,1)==8 ,:);
s = R(:,1);

% Map digit -> class : {2,8} -> {1,-1}
R = R(:,2:end)';
s(s(:)==2) = 1;
s(s(:)==8) = -1;

[d2,M] = size(R);
percepClass = repmat(s',maxIter,1) .* (weights' * [ones(1,M);R]);
testError = sum(percepClass < 0, 2)/M;

x = 1:maxIter;
figure(1)
plot(x,trainingError, '-o', 'Linewidth', 1.5)
hold on
plot(x,testError, '-o', 'Linewidth', 1.5)

% Figure Options
title('Training, Test Errors vs Iterations (PerceptronLR; Feature Vectors)','FontSize',46);
xlabel('Iterations','FontSize',36);
ylabel('Error Probability','FontSize',36);
legend('Training Error','Test Error');
grid on
grid minor
set(gca,'fontsize',32);
xlim([0 500]);
%axis([floor(min(X(1,:))) ceil(max(X(1,:))) floor(min(X(2,:))) ceil(max(X(2,:)))])
hold off


