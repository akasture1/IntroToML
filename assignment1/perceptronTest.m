% Simple first test script for the perceptron algorithm

% Generate (x,y) pairs
N = 100;
X = zeros(2,N);
y = zeros(N,1);
c = 0.1;

i=0;
while i ~= N
    x = rand(2,1);
    if( x(2) < x(1) - c ) 
        sgn = -1; cl = 'r*';
        i = i + 1;
        X(:,i) = x; y(i) = sgn;
        plot(X(1,i),X(2,i), cl);
        hold on;
    elseif ( x(2) > x(1) + c )
        sgn = 1; cl = 'b*';   
        i = i + 1;
        X(:,i) = x; y(i) = sgn;
        plot(X(1,i),X(2,i), cl);
        hold on;
    end
end

% find optimal weights
[w, iter] = perceptron(X,y)

% plot line defined by the vector w
t = 0:0.01:1;
plot(t, -(w(1)/w(3)) - (t * w(2)/w(3)), 'k' ,'Linewidth', 2.5);

% figure options
title('The Perceptron Learning Algorithm','FontSize',24);
xlabel('x_{1}','FontSize',24);
ylabel('x_{2}','FontSize',24);
grid on
grid minor
set(gca,'fontsize',20);
xlim([0 1]);ylim([0 1])
