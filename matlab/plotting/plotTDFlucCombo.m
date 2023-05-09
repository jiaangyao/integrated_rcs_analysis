function [figTD_LDCombo, fFigure, outputStruct] = ...
    plotTDFlucCombo(timePowLD0, powDataLD0, timePowLD1, powDataLD1, ...
    timeLD, LDData_LDCombo, LDData_LD0, LDData_LD1, LDThresh, ...
    timeState, stateData, currentData, ... % timeAWTre, AWTremData, ...
    timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
    motorDiaryInterp, motorDiary, ...
    medTime, vecDyskOnset, vecDyskOffset, strLDCombo, ...
    strLD0, strCh_LD0, strPB_LD0, ...
    strLD1, strCh_LD1, strPB_LD1, ...
    cfg, varargin)
% powData - single channel data
% LDData - single channel data

%% Input parsing
% adding custom input flag so can implement new path searching without
% modifying existing scripts - JY 07/18/2022

% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'boolStreamed_LD0', true, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'boolStreamed_LD1', true, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));

addParameter(p, 'boolPlotSmooth', true, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'lenSmoothSec', 20, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'origAlpha', 0.3, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'smoothAlpha', 1, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

% color arguments
addParameter(p, 'colorLDCombo', [0, 0.4470, 0.7410], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'colorLD0', [0, 0.4470, 0.7410], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'colorLD1', [0.8500, 0.3250, 0.0980], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'smoothColor', [1, 0, 0], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

% plotting limits
addParameter(p, 'ylimPowLD0', [0, 2000], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimPowLD1', [0, 300], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

addParameter(p, 'ylimLD_LDCombo', [-300, 2000], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimLD_LD0', [0, 500], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimLD_LD1', [0, 100], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

addParameter(p, 'ylimState', [-0.5, 1.5], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimCurrent', [2.8, 3.6], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimStateTrans', [-100, 800], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimAWProb', [-0.2, 1.2], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimPKGBrady', [-5, 155], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimPKGDysk', [-1.1, 3.1], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimMDRange', [-0.5, 3.5], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

% output parameters
addParameter(p, 'boolReturnOutput', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'outputStruct', NaN);

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
boolStreamed_LD0 = p.Results.boolStreamed_LD0;
boolStreamed_LD1 = p.Results.boolStreamed_LD1;

boolPlotSmooth = p.Results.boolPlotSmooth;
lenSmoothSec = p.Results.lenSmoothSec;
origAlpha = p.Results.origAlpha;
smoothAlpha = p.Results.smoothAlpha;

colorLDCombo = p.Results.colorLDCombo;
colorLD0 = p.Results.colorLD0;
colorLD1 = p.Results.colorLD1;
smoothColor = p.Results.smoothColor;

ylimPowLD0 = p.Results.ylimPowLD0;
ylimPowLD1 = p.Results.ylimPowLD1;

ylimLD_LDCombo = p.Results.ylimLD_LDCombo;
ylimLD_LD0 = p.Results.ylimLD_LD0;
ylimLD_LD1 = p.Results.ylimLD_LD1;

ylimState = p.Results.ylimState;
ylimCurrent = p.Results.ylimCurrent;
ylimStateTrans = p.Results.ylimStateTrans;
ylimAWProb = p.Results.ylimAWProb;
ylimPKGBrady = p.Results.ylimPKGBrady;
ylimPKGDysk = p.Results.ylimPKGDysk;
ylimMDRange = p.Results.ylimMDRange;

boolReturnOutput = p.Results.boolReturnOutput;
outputStruct = p.Results.outputStruct;

% define any magic numbers
TIME_RES_MD_IN_MINUTES = 30;

% obtain the side of the recording
strSide = cfg.vec_str_side{1}(1);

if strcmp(cfg.str_sub, 'RCS02')
    strMDBradyLabel = 'MD Brady Intensity';
    strMDDyskLabel = 'MD Dysk Intensity';
    strAWDispLabel = 'AW Dysk Prob';

    if strcmp(strSide, 'L')
        strMDBrady = 'RSlowness';
        strMDBradyTbsome = 'RSlownessTbsome';
        
        strMDDysk = 'LDysk';
        strMDDyskTbsome = 'LDyskTbsome';
        
    elseif strcmp(strSide, 'R')
        strMDBrady = 'LSlowness';
        strMDBradyTbsome = 'LSlownessTbsome';

        strMDDysk = 'LDysk';
        strMDDyskTbsome = 'LDyskTbsome';
    else
        error('Unknown side');
    end

elseif strcmp(cfg.str_sub, 'RCS14')
    ylimAWProb = [-1, 6];
    strMDBradyLabel = 'MD Trem Intensity';
    strMDDyskLabel = 'MD Dysk Intensity';
    strAWDispLabel = 'AW Acceleration';

    strMDBrady = 'RHandTremor';
    strMDBradyTbsome = 'RHandTremorTbsome';
    
    strMDDysk = 'Dyskinesia';
    strMDDyskTbsome = 'DyskinesiaTbsome';
else
    error('Unknown subject')
end

% set various font sizes
titleFontSize = 16;
labelFontSize = 16;
legendFontSize = 13;
tickFontSize = 13;


%% set up figure


% determine if AW and PKG data are valid
boolAWValid = ~any(strcmp(cfg.str_no_aw_data_day, cfg.str_data_day));
boolPKGValid = ~any(strcmp(cfg.str_no_pkg_data_day, cfg.str_data_day));

if boolAWValid && boolPKGValid
    subplotNum = 9; % both exists
    figTD_LDCombo = figure('position', [10, 10, 2600, 2000]); 

elseif boolAWValid || boolPKGValid
    subplotNum = 8; % one of them exists
    figTD_LDCombo = figure('position', [10, 10, 2500, 1900]); 
else
    subplotNum = 7;
    figTD_LDCombo = figure('position', [10, 10, 2400, 1800]); 
end

idxCurrSubplot = 1;

%% plotting the raw power data

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;

if ~boolPlotSmooth
    % first plot LD0 power using the left y-axis
    yyaxis left;
    plot(timePowLD0, powDataLD0, 'Color', colorLD0, 'DisplayName', ...
        strcat(strCh_LD0,' | ', strPB_LD0));

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize; 
    
    % add label
    ylabel('Power (A.U.)', 'FontSize', labelFontSize)
    ylim(ylimPowLD0)
    
    % next plot LD1 power using right y-axis
    yyaxis right;
    plot(timePowLD1, powDataLD1, 'Color', colorLD1, 'DisplayName', ...
        strcat(strCh_LD1,' | ', strPB_LD1));

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize; 
    
    % add label
    ylabel('Power (A.U.)', 'FontSize', labelFontSize)
    ylim(ylimPowLD1)
    
    % add legend
    lgdCurr = legend('boxoff');
    lgdCurr.FontSize = legendFontSize;

else
    % first plot LD0 power using the left y-axis
    % plot the original unsmoothed data
    yyaxis left;
    plot(timePowLD0, powDataLD0, 'Color', [colorLD0, origAlpha], ...
        'LineWidth', 0.5, 'HandleVisibility', 'off');

    % now compute the moving average
    fsPowLD0Curr = seconds(mode(diff(timePowLD0)));
    powDataLD0Smooth = movmean(powDataLD0, ...
        round(1/fsPowLD0Curr * lenSmoothSec));
    plot(timePowLD0, powDataLD0Smooth, 'Color', [colorLD0, smoothAlpha], ...
        'LineStyle', '-', 'LineWidth', 1, 'DisplayName', ...
        strcat(strCh_LD0,' | ', strPB_LD0))

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize; 
    
    % add label
    ylabel('Power (A.U.)', 'FontSize', labelFontSize)
    ylim(ylimPowLD0)

    % next plot LD1 power using the right y-axis
    yyaxis right;
    plot(timePowLD1, powDataLD1, 'Color', [colorLD1, origAlpha], ...
        'LineWidth', 0.5, 'HandleVisibility', 'off');
    
    fsPowLD1Curr = seconds(mode(diff(timePowLD1)));
    powDataLD1Smooth = movmean(powDataLD1, ...
        round(1/fsPowLD1Curr * lenSmoothSec));
    plot(timePowLD1, powDataLD1Smooth, 'Color', [colorLD1, smoothAlpha], ...
        'LineStyle', '-', 'LineWidth', 1, 'DisplayName', ...
        strcat(strCh_LD1,' | ', strPB_LD1))

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize; 
    
    % add label
    ylabel('Power (A.U.)', 'FontSize', labelFontSize)
    ylim(ylimPowLD1)
        
    % add legend
    lgdCurr = legend('boxoff');
    lgdCurr.FontSize = legendFontSize;
end

% plot times for when med was taken
for i = 1:numel(medTime)
    xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
end

% add title
title(sprintf("Power band streamed"), 'FontSize', titleFontSize)

idxCurrSubplot = idxCurrSubplot + 1;

%% plot the LD data

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;

if ~boolPlotSmooth
    % no smoothing is performed
    % first plot LD0 data using the left y-axis
    yyaxis left;
    plot(timeLD, LDData_LD0, 'Color', colorLD0, 'DisplayName', strLD0);
        
    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize; 
    
    % add label
    ylabel("LD Value (A.U.)", 'FontSize', labelFontSize)
    ylim(ylimLD_LD0)
    
    % next plot LD1 data using the right y-axis
    yyaxis right;
    plot(timeLD, LDData_LD1, 'Color', colorLD1, 'DisplayName', strLD1);

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;
    
    % add label
    ylabel("LD Value (A.U.)", 'FontSize', labelFontSize)
    ylim(ylimLD_LD1)
    
    % add legend
    lgdCurr = legend('boxoff');
    lgdCurr.FontSize = legendFontSize;

else
    % first plot LD0 using the left y-axis
    % plot the original data
    yyaxis left;
    plot(timeLD, LDData_LD0, 'Color', [colorLD0, origAlpha], ...
        'LineWidth', 0.5, 'HandleVisibility','off');
    fsLDCurr = seconds(mode(diff(timeLD)));

    if numel(LDData_LD0) > 1
        % now compute the moving average
        LDData_LD0Smooth = movmean(LDData_LD0, ...
            round(1/fsLDCurr * lenSmoothSec));
        plot(timeLD, LDData_LD0Smooth, 'Color', [colorLD0, smoothAlpha], ...
            'LineStyle', '-', 'LineWidth', 1, 'DisplayName', strLD0)
    end

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;

    % add label
    ylabel("LD Value (A.U.)", 'FontSize', labelFontSize)
    ylim(ylimLD_LD0)
    
    % now plot for LD1
    yyaxis right;
    plot(timeLD, LDData_LD1, 'Color', [colorLD1, origAlpha], ...
        'LineWidth', 0.5, 'HandleVisibility','off');

    if numel(LDData_LD1) > 1
        % now compute the moving average
        LDData_LD1Smooth = movmean(LDData_LD1, ...
            round(1/fsLDCurr * lenSmoothSec));
        plot(timeLD, LDData_LD1Smooth, 'Color', [colorLD1, smoothAlpha], ...
            'LineStyle', '-', 'LineWidth', 1, 'DisplayName', strLD1)
    end

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;

    % add label
    ylabel("LD Value (A.U.)", 'FontSize', labelFontSize)
    ylim(ylimLD_LD1)

    % add legend
    lgdCurr = legend('boxoff');
    lgdCurr.FontSize = legendFontSize;
end

% plot times for when med was taken
for i = 1:numel(medTime)
    xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
end

% add title
if boolStreamed_LD0 && boolStreamed_LD1
    title(sprintf("Individual LDs Streamed"), "FontSize", titleFontSize)
else
    title(sprintf("Individual LDs Estimated Offline"), 'FontSize', titleFontSize)
end

idxCurrSubplot = idxCurrSubplot + 1;


%% now plot the combo

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;

% first preprocess 
LDData_LDCombo_NeedFix = LDData_LDCombo(LDData_LDCombo >= 2^31);
LDData_LDCombo(LDData_LDCombo >= 2^31) = LDData_LDCombo_NeedFix - 2^32;

if ~boolPlotSmooth
    % plot the unsmoothed data
    plot(timeLD, LDData_LDCombo, 'Color', colorLDCombo, ....
        'HandleVisibility','off');

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;

else
    % plot the original data
    plot(timeLD, LDData_LDCombo, 'Color', [colorLDCombo, origAlpha], ...
        'DisplayName','Unsmoothed');

    % now compute the moving average
    fsLDCurr = seconds(mode(diff(timeLD)));
    LDData_LDComboSmooth = movmean(LDData_LDCombo, ...
        round(1/fsLDCurr * lenSmoothSec));
    plot(timeLD, LDData_LDComboSmooth, 'Color', [smoothColor, smoothAlpha], ...
        'LineStyle', '-', 'LineWidth', 1, 'DisplayName','Smoothed')
    
    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;
    
    % add legend
    lgdCurr = legend('boxoff');
    lgdCurr.FontSize = legendFontSize;
end

% plot times for when med was taken
for i = 1:numel(medTime)
    xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
end

% plot the threshold for the LD
yline(LDThresh, '--', 'Color', 'g', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('%s threshold', strLDCombo));

% add label
ylabel("LD Value (A.U.)")
ylim(ylimLD_LDCombo);

% add title
if boolStreamed_LD0 && boolStreamed_LD1
    title(sprintf("%s Streamed", strLDCombo))
else
    title(sprintf("%s Estimated Offline", strLDCombo))
end

idxCurrSubplot = idxCurrSubplot + 1;


%%  plot the states

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;

if ~boolPlotSmooth
    % plot the unsmoothed data
    plot(timeState, stateData, 'Color', colorLDCombo)

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;

else
    % plot the original data
    plot(timeState, stateData, 'Color', [colorLDCombo, origAlpha]);

    % now compute the moving average
    if numel(stateData) > 1
        fsAdapCurr = seconds(mode(diff(timeState)));
        stateDataSmooth = movmean(stateData, round(1/fsAdapCurr * lenSmoothSec));
        plot(timeState, stateDataSmooth, 'Color', [smoothColor, smoothAlpha], ...
            'LineWidth', 0.5, 'DisplayName', 'Smoothed')
    end

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;
end

% plot times for when med was taken
for i = 1:numel(medTime)
    xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
end

% add label
ylabel("Decoder State", 'FontSize', labelFontSize)
ylim(ylimState)

% add title
if boolStreamed_LD0 && boolStreamed_LD1
    title('Decoder State Streamed', 'FontSize', titleFontSize)
else
    title("Decoder State Estimated Offline", 'FontSize', titleFontSize)
end

idxCurrSubplot = idxCurrSubplot + 1;

%% plot the current injected

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
if ~boolPlotSmooth
    % plot the unsmoothed data
    plot(timeState, currentData, 'Color', colorLDCombo)

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;

else
    % plot the original data
    plot(timeState, currentData, 'Color', [colorLDCombo, origAlpha]);
    
    if numel(currentData) > 1
        % now compute the moving average
        currentDataSmooth = movmean(currentData, round(1/fsAdapCurr * lenSmoothSec));
        plot(timeState, currentDataSmooth, 'Color', [smoothColor, smoothAlpha], ...
            'LineWidth', 0.5, 'HandleVisibility', 'off')
    end

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;

end

% plot times for when med was taken
for i = 1:numel(medTime)
    xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
end

% add label
ylabel("Current Injected (mA)", 'FontSize', labelFontSize)
ylim(ylimCurrent)

% add title
if boolStreamed_LD0 && boolStreamed_LD1
    title('Current Injected Streamed', 'FontSize', titleFontSize)
else
    title('Current Injected Estimated', 'FontSize', titleFontSize)
end

idxCurrSubplot = idxCurrSubplot + 1;


%% estimate the average speed of state transition

% compute sample diff and figure out rate of change
if numel(stateData) > 1
    vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;

    stateDataDiff = diff(stateData);
    idxStateFull = 1:numel(stateDataDiff);
    
    if ~boolPlotSmooth
        fsAdapCurr = seconds(mode(diff(timeState)));
    end
    idxChange = idxStateFull(stateDataDiff ~=0);
    if numel(idxChange) ~= 0
        stateChangeRate = diff([1, idxChange]) * fsAdapCurr;
        timeChangeRate = timeState(idxChange);
    else
        timeChangeRate = timeState;
        stateChangeRate = zeros(size(timeChangeRate));
    end
    
    % now generate the plot
    plot(timeChangeRate, stateChangeRate)
    for i = 1:numel(medTime)
        xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
    end

    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;
    
    % add label
    ylabel("Time (s)", 'FontSize', labelFontSize);
    ylim(ylimStateTrans);
    
    % add title
    title("Time between State Transition", 'FontSize', titleFontSize)
    
    idxCurrSubplot = idxCurrSubplot + 1;

end

%% now plot the Apple Watch data if it exists

if ~any(strcmp(cfg.str_no_aw_data_day, cfg.str_data_day))
    vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
    
    % plot the dysk prob
    plot(timeAWDysk, AWDyskData, 'DisplayName', strAWDispLabel, ...
        'LineWidth', 2);
    
    % change tick size
    ax = gca;
    ax.FontSize = tickFontSize;
    
    % add label
    ylabel("Probability", 'FontSize', labelFontSize)
    ylim(ylimAWProb)
    
    % add title
    title("Apple Watch data", 'FontSize', titleFontSize)
    
    % add legend
    lgdCurr = legend('boxoff');
    lgdCurr.FontSize = legendFontSize;

    idxCurrSubplot = idxCurrSubplot + 1;
end

%% next plot the PKG data if it exists

if ~any(strcmp(cfg.str_no_pkg_data_day, cfg.str_data_day))
    vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;

    % plot the brady scores
    yyaxis left;
    plot(timePKG, PKGBradyData, 'LineWidth', 2, ...
        'DisplayName', 'PKG Brady Score');
    yline(25, 'Color', [0, 0.4470, 0.7410], 'LineStyle', '--', ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');
    ax = gca;
    ax.FontSize = tickFontSize;
    ylabel("Brady score (a.u.)", 'FontSize', labelFontSize)
    ylim(ylimPKGBrady)
    
    % plot the dysk scores
    yyaxis right;
    plot(timePKG, log10(PKGDyskData), 'LineWidth', 2, ...
        'DisplayName', 'PKG Dysk Score');
    yline(log10(10), 'Color', [0.8500, 0.3250, 0.0980], 'LineStyle', '--', ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');
    ax = gca;
    ax.FontSize = tickFontSize;
    ylabel("Log Dysk score (a.u.)", 'FontSize', labelFontSize)
    ylim(ylimPKGDysk)

    title("PKG data", 'FontSize', titleFontSize)
    lgdCurr = legend('boxoff');
    lgdCurr.FontSize = legendFontSize;
    idxCurrSubplot = idxCurrSubplot + 1;
end
 

%% now plot the motor diary dysk

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;

% plot the brady and dysk intensities
plot(motorDiaryInterp.time, motorDiaryInterp.(strMDDysk), ...
    "DisplayName", strMDDyskLabel, 'LineWidth', 2);
plot(motorDiaryInterp.time, motorDiaryInterp.(strMDBrady), ...
    "DisplayName", strMDBradyLabel, 'LineWidth', 2);

% plot times for when med was taken
for i = 1:numel(medTime)
    xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
end

% change tick size
ax = gca;   
ax.FontSize = tickFontSize;

% add label
ylabel('MD Intensity', 'FontSize', labelFontSize)
ylim(ylimMDRange)

% plot the dyskinesia labels
for i = 1:numel(vecDyskOnset)
    if i == 1
        xline(vecDyskOnset(i), 'r', 'LineWidth', 2, ...
            'DisplayName','Self-reported Dysk Start')
    else
        xline(vecDyskOnset(i), 'r', 'LineWidth', 2, ...
            'HandleVisibility','off')
    end
end

for i = 1:numel(vecDyskOffset)
    if i == 1
        xline(vecDyskOffset(i), 'r--', 'LineWidth', 2, ...
            'DisplayName','Self-reported Dysk End')
    else
        xline(vecDyskOffset(i), 'r--', 'LineWidth', 2, ...
            'HandleVisibility','off')
    end
end

% change the range of the plot
ylimRangeCurr = ylimMDRange;

% now fill in the motor diary ON/OFF times
% now fill in all the on time
idxON = (motorDiary.ON == 1);
onRedacted = motorDiary(idxON, :);
fillMDRedacted(onRedacted, ylimRangeCurr, 'g', 'ON Period', ...
    TIME_RES_MD_IN_MINUTES)

% now fill in all the off time
idxOFF = (motorDiary.OFF == 1);
offRedacted = motorDiary(idxOFF, :);
fillMDRedacted(offRedacted, ylimRangeCurr, 'r', 'OFF Period', ...
    TIME_RES_MD_IN_MINUTES)

% add title
title('Motor Diary Data', 'FontSize', titleFontSize)

% add legend
lgdCurr = legend('boxoff');
lgdCurr.FontSize = legendFontSize;

%% saving output figure

linkaxes([vecAxes{:}], 'x');
datetick('x','HH:MM');

% save the output figure
fFigure = sprintf('%s_%s_%s_Fluc_%s_%s', cfg.str_sub, ...
    cfg.str_aDBS_paradigm(1:5), 'LDCombo', cfg.str_data_day, ...
    cfg.vec_str_side{1}(1));


%% specify the output struct and storing all useful information

if boolReturnOutput
    % compute the total amount of time spent in on and off
    perMD_ON = sum(motorDiary.ON) / 24;
    perMD_OFF = sum(motorDiary.OFF) / 24;

    % compute the total amount of time spent in brady and dysk
    perMDBrady = sum(motorDiary.LSlowness > 0) / 24;
    perMDBradyTbsome = sum(motorDiary.LSlownessTbsome > 0) / 24;
    perMDDysk = sum(motorDiary.LDysk > 0) / 24;
    perMDDyskTbsome = sum(motorDiary.LDysk > 0) / 24;
    
    % concatenate all non-zero brady and dysk intensity
    idxMDBrady = motorDiary.LSlowness > 0;
    if sum(idxMDBrady) > 0
        allMDBrady = motorDiary.LSlowness(idxMDBrady);
    else
        allMDBrady = [];
    end

    idxMDDysk = motorDiary.LDysk > 0;
    if sum(idxMDDysk) > 0
        allMDDysk = motorDiary.LDysk(idxMDDysk);
    else
        allMDDysk = [];
    end

    % append to output
    outputStruct.perMD_ON = perMD_ON;
    outputStruct.perMD_OFF = perMD_OFF;

    outputStruct.perMDBrady = perMDBrady;
    outputStruct.perMDBradyTbsome = perMDBradyTbsome;
    outputStruct.perMDDysk = perMDDysk;
    outputStruct.perMDDyskTbsome = perMDDyskTbsome;

    outputStruct.allMDBrady = allMDBrady;
    outputStruct.allMDDysk = allMDDysk;

    % compute statistics with Apple watch data if exists
    if ~any(strcmp(cfg.str_no_aw_data_day, cfg.str_data_day))
        idxValidAWDysk = AWDyskData > 0;
        perAWDysk = sum(idxValidAWDysk) / (24 * 60);
        allAWDysk = AWDyskData;
        avgAWDysk = mean(AWDyskData);
        
        onAWDysk = [];
        % extract all data during the ON period
        for i = 1:size(onRedacted)
            currTimeEnd = onRedacted.time(i);
            currTimeStart = currTimeEnd - minutes(TIME_RES_MD_IN_MINUTES);

            idxCurrTime = timeAWDysk >= currTimeStart & ...
                timeAWDysk < currTimeEnd;
            currAwDysk = AWDyskData(idxCurrTime);
            onAWDysk = [onAWDysk; currAwDysk];   
        end

        offAWDysk = [];
        % extract all data during the OFF period
        for i = 1:size(offRedacted)
            currTimeEnd = offRedacted.time(i);
            currTimeStart = currTimeEnd - minutes(TIME_RES_MD_IN_MINUTES);

            idxCurrTime = timeAWDysk >= currTimeStart & ...
                timeAWDysk < currTimeEnd;
            currAwDysk = AWDyskData(idxCurrTime);
            offAWDysk = [offAWDysk; currAwDysk];   
        end
        
        outputStruct.perAWDysk = perAWDysk;
        outputStruct.allAWDysk = allAWDysk;
        outputStruct.avgAWDysk = avgAWDysk;
        
        outputStruct.onAWDysk = onAWDysk;
        outputStruct.offAWDysk = offAWDysk;
    end

    % compute statistics with PKG data if exists
    if ~any(strcmp(cfg.str_no_pkg_data_day, cfg.str_data_day))
        allPKGBrady = PKGBradyData;
        allPKGDysk = PKGDyskData;
        
        onPKGBrady = [];
        onPKGDysk = [];
        % extract all data during the ON period
        for i = 1:size(onRedacted)
            currTimeEnd = onRedacted.time(i);
            currTimeStart = currTimeEnd - minutes(TIME_RES_MD_IN_MINUTES);

            idxCurrTime = timePKG >= currTimeStart & ...
                timePKG < currTimeEnd;
            currPKGBrady = PKGBradyData(idxCurrTime);
            currPKGDysk = PKGDyskData(idxCurrTime);

            onPKGBrady = [onPKGBrady; currPKGBrady];   
            onPKGDysk = [onPKGDysk; currPKGDysk];
        end

        offPKGBrady = [];
        offPKGDysk = [];
        % extract all data during the OFF period
        for i = 1:size(offRedacted)
            currTimeEnd = offRedacted.time(i);
            currTimeStart = currTimeEnd - minutes(TIME_RES_MD_IN_MINUTES);

            idxCurrTime = timePKG >= currTimeStart & ...
                timePKG < currTimeEnd;
            currPKGBrady = PKGBradyData(idxCurrTime);
            currPKGDysk = PKGDyskData(idxCurrTime);

            offPKGBrady = [offPKGBrady; currPKGBrady];   
            offPKGDysk = [offPKGDysk; currPKGDysk];
        end

        outputStruct.allPKGBrady = allPKGBrady;
        outputStruct.allPKGDysk = allPKGDysk;

        outputStruct.onPKGBrady = onPKGBrady;
        outputStruct.onPKGDysk = onPKGDysk;

        outputStruct.offPKGBrady = offPKGBrady;
        outputStruct.offPKGDysk = offPKGDysk;
    end
    
end

end

function fillMDRedacted(MDRedacted, ylimMDRange, fillColor, strLabel, ...
    TIME_RES_MD_IN_MINUTES)

for i = 1:size(MDRedacted, 1)
    endTime = MDRedacted.time(i);
    startTime = endTime - minutes(TIME_RES_MD_IN_MINUTES);

    rectX = [startTime, endTime];
    rectY = [min(ylimMDRange), max(ylimMDRange)];
    


    % now fill in
    if i == 1
        patch(rectX([1, 2, 2, 1]), rectY([1, 1, 2, 2]), fillColor, ...
            'facealpha', 0.1, 'edgecolor', 'none',  ...
            "DisplayName", strLabel);
    else
        patch(rectX([1, 2, 2, 1]), rectY([1, 1, 2, 2]), fillColor, ...
            'facealpha', 0.1, 'edgecolor', 'none',  ...
            "HandleVisibility", 'off');

%         area([startTime,endTime], [min(ylimMDRange), max(ylimMDRange)], ...
%             'facecolor', fillColor, 'facealpha', 0.1, ...
%             'edgecolor', 'none', 'basevalue', 0, ...
%             "HandleVisibility", "off");
    end

end

end