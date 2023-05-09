    figure; ax1 = subplot(411); hold on;
    plot(timeRaw,rawData(:, 1), 'LineWidth', 2);

    ax2 = subplot(412); hold on;
    plot(timePowLD0, powDataLD0, 'LineWidth', 2); 

    ax3 = subplot(413); hold on;
    plot(timeStateLD0, stateDataLD0, 'LineWidth', 2);

    ax4 = subplot(414); hold on;
    plot(timeStateLD0, currentDataLD0, 'LineWidth', 2);
    linkaxes([ax1, ax2, ax3, ax4], 'x')