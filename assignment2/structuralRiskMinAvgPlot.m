figure
avgSrmTrainErrors = mean(srmTrainErrors,2);
avgSrmTestErrors = mean(srmTestErrors,2);

loglog(trainData,avgSrmTrainErrors, 'x-', 'MarkerSize', 15, 'LineWidth', 4)
hold on
loglog(trainData,avgSrmTestErrors, 'x-', 'MarkerSize', 15, 'LineWidth', 4)

% Figure options
title('SRM Training vs Test Error; 0.01\Omega','FontSize',46);
xlabel('Number of Training Points','FontSize',36);
ylabel('Error','FontSize',36);
legend('Training Error','Test Error');
grid on
grid minor
set(gca,'fontsize',32);
    
for j = 1:qMax
    figure
    avgErmTrainErrors = mean(ermTrainErrors,3);
    avgErmTestErrors = mean(ermTestErrors,3);

    loglog(trainData,avgErmTrainErrors(:,j), 'x-', 'MarkerSize', 15, 'LineWidth', 4)
    hold on
    loglog(trainData,avgErmTestErrors(:,j), 'x-', 'MarkerSize', 15, 'LineWidth', 4)

    % Figure options
    title(['ERM Training vs Test Error; Q=' num2str(j-1) ''],'FontSize',46);
    xlabel('Number of Training Points','FontSize',36);
    ylabel('Error','FontSize',36);
    legend('Training Error','Test Error');
    grid on
    grid minor
    set(gca,'fontsize',32);
end

