    figure; ax1 = subplot(511); hold on;
    LD_Estimated = 6*ld0_data_valid_sorted(:, 1) - ...
        14*ld0_data_valid_sorted(:, 2);

    fsLD = seconds(mode(diff(timeLD_LDCombo)));
    LD_EstimatedSmoothed = movmean(LD_Estimated, round(1/fsLD * 20));

    plot(timeLD_LDCombo, LD_Estimated, 'Color', [0, 0.4470, 0.7410, 0.3]); 
    plot(timeLD_LDCombo, LD_EstimatedSmoothed, 'Color', [1, 0, 0])
    ylim([-500, 5000])

    ax2 = subplot(512); hold on;
    LDData_LDCombo_Smoothed = movmean(LDData_LDCombo, round(1/fsLD * 20));
    plot(timeLD_LDCombo, LDData_LDCombo, 'Color', [0, 0.4470, 0.7410, 0.3]);
    plot(timeLD_LDCombo, LDData_LDCombo_Smoothed, 'Color', [1, 0, 0]); 
    ylim([-500, 5000])
    yline(LDThresh_LDCombo, '--', 'Color', 'g', 'LineWidth', 2.5);

%     ax3 = subplot(513); hold on;
%     vecLDComboValidSortedOrigSmoothed = ...
%         movmean(vecLDComboValidSortedOrig, round(1/fsLD * 20));
%     plot(timeLD_LDCombo, vecLDComboValidSortedOrig, ...
%         'Color', [0, 0.4470, 0.7410, 0.3]); 
%     plot(timeLD_LDCombo, vecLDComboValidSortedOrigSmoothed, ...
%         'Color', [1, 0, 0]); 
%     ylim([-500, 5000])
%     yline(LDThresh_LDCombo, '--', 'Color', 'g', 'LineWidth', 2.5);
    ax3 = subplot(513); hold on;
    fsState = seconds(mode(diff(timeStateLDCombo)));
    stateDataLDComboOrigSmooth = movmean(state_valid_sorted_remove14, round(1/fsState * 20));
    plot(timeStateLDCombo, state_valid_sorted_remove14, 'Color', [0, 0.4470, 0.7410, 0.3]);
    plot(timeStateLDCombo, stateDataLDComboOrigSmooth, 'Color', [1, 0, 0]);
    ylim([-1.5, 3.5]);

    ax4 = subplot(514); hold on;
    stateDataLDComboSmooth = movmean(stateDataLDCombo, round(1/fsState * 20));
    plot(timeStateLDCombo, stateDataLDCombo, 'Color', [0, 0.4470, 0.7410, 0.3]);
    plot(timeStateLDCombo, stateDataLDComboSmooth, 'Color', [1, 0, 0]);
    ylim([-0.5, 1.5]);
    
    ax5 = subplot(515); hold on;
    currentDataLDComboSmooth = movmean(currentDataLDCombo, ...
        round(1/fsState * 20));
    plot(timeStateLDCombo, currentDataLDCombo, ...
        'Color', [0, 0.4470, 0.7410, 0.3]);
    plot(timeStateLDCombo, currentDataLDComboSmooth, 'Color', [1, 0, 0])
    ylim([2.7, 3.7]);
    linkaxes([ax1, ax2, ax3, ax4, ax5], 'x');