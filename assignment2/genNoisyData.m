% Assignment 2 - Problem 5
% Noisy Classification Problem
% x1: [0,2.5]; x2: [-1,2]

f = @(t) t.*(t-1).*(t-2);

n = 50;
x1 = 2.5*rand(n,1); 
x2 = -1 + 3*rand(n,1);
y = zeros(n,1);

y(x2 >= f(x1)) = 1;
y(x2 < f(x1)) = -1;

[~,idxFlip] = datasample(y,floor(0.1*n),1,'Replace',false);
y(idxFlip) = -1 * y(idxFlip);

classAbove = find(y==1);
classBelow = find(y==-1);

figure(1);
t = 0:0.01:2.5;
p1 = plot(t,f(t),'LineWidth', 5);
hold on
p2 = plot(x1(classAbove),x2(classAbove), 'x', 'MarkerSize', 15, 'LineWidth', 2);
p3 = plot(x1(classBelow),x2(classBelow), 'o', 'MarkerSize', 15, 'LineWidth', 2);

% Figure options
title('Generating Noisy Data','FontSize',46);
xlabel('x_{1}','FontSize',36);
ylabel('x_{2}','FontSize',36);
%legend([p1 p3],'true separators','perceptron separator','Location','southeast');
grid on
grid minor
set(gca,'fontsize',32);
xlim([0 2.5]);ylim([-1 2])
