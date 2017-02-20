% Assignment 2 
% Problem 2

pert = 4e-2;
x = 0.8*rand(100,1) + 0.1;
mid1 = median(x) + pert;
mid2 = median(x) - pert;
high = max(x) - pert;
low = min(x) + pert;

%nsew
x1 = [mid1;mid2;high;low];
x2 = [high;low;mid1;mid2];

figure(1)
p1 = plot(x1, x2, 'x', 'MarkerSize', 20, 'LineWidth', 5);
hold on
p2 = plot(mean(x1), mean(x2), 'o', 'MarkerSize', 20, 'LineWidth', 5);

rectangle('Position',[x1(4)-pert x2(2)-pert x1(3)-x1(4)+2*pert x2(1)-x2(2)+2*pert])

% Figure options
title('Axis-aligned rectangles VC Dimension Illustration','FontSize',46);
xlabel('x_{1}','FontSize',36);
ylabel('x_{2}','FontSize',36);
legend([p1 p2],'+1','-1','Location','southeast');
grid on
%grid minor
set(gca,'fontsize',32);
xlim([0 1]);ylim([0 1])


