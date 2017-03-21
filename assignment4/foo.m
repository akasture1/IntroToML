clear
close all
rng('default')

kScale = 10;

% Initialise parameters
N = 50;
X = zeros(2,N);
y = zeros(N,1);
c = 0.1;
gamma = 0.1;
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
XBar = X';

svmObj = fitcsvm(XBar,y,'Standardize',0,'BoxConstraint', Inf);

t = 0:0.01:1;
% Plot Training Data
figure(1)
classAbove = find(y== 1);
classBelow = find(y==-1);
plot(X(1,classAbove), X(2,classAbove), 'x', 'MarkerSize', 5, 'LineWidth', 1);
hold on
plot(X(1,classBelow), X(2,classBelow), 'o', 'MarkerSize', 5, 'LineWidth', 1);

p1 = plot(t, t + c + gamma, 'k','Linewidth', 2);                            % true separator
p2 = plot(t, t + c - gamma, 'k' ,'Linewidth', 2);      

d=0.01;
[x1Grid, x2Grid] = meshgrid(min(XBar(:,1)):d:max(XBar(:,1)), min(XBar(:,2)):d:max(XBar(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
[~,scores] = predict(svmObj, xGrid);

plot(XBar(svmObj.IsSupportVector,1),XBar(svmObj.IsSupportVector,2),'ko','MarkerSize',7,'LineWidth', 2)
contour(x1Grid,x2Grid, reshape(scores(:,2), size(x1Grid)), [0,0], 'LineWidth', 2);

title(['The Perceptron Learning Algorithm;    \gamma =' num2str(gamma) ''],'FontSize',46);
xlabel('x_{1}','FontSize',36);
ylabel('x_{2}','FontSize',36);
%legend([p1 p3],'true separators','perceptron separator','Location','southeast');
grid on
grid minor
set(gca,'fontsize',32);
xlim([0 1]);ylim([0 1])