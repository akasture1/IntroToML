figure
d=0.01;
[x1Grid, x2Grid] = meshgrid(min(XBar(:,1)):d:max(XBar(:,1)), min(XBar(:,2)):d:max(XBar(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];
[~,scores] = predict(svmObj, xGrid);

plot(XBar((y==-1),1), XBar((y==-1),2), 'x', 'MarkerSize', 5, 'LineWidth', 1);
hold on
plot(XBar((y==1),1), XBar((y==1),2), 'o', 'MarkerSize', 5, 'LineWidth', 1);
plot(sv(:,1),sv(:,2),'ko','MarkerSize',7,'LineWidth', 2)
contour(x1Grid,x2Grid, reshape(scores(:,2), size(x1Grid)), [0,0], 'LineWidth', 2);
grid on
grid minor