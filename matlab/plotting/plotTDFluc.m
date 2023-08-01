function [figTD_LD, fFigure, outputStruct] = ...
    plotTDFluc(timePow, powData, timeLD, LDData, ...
    timeState, stateData, currentData, ... % timeAWTre, AWTremData, ...
    timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData,...
    motorDiaryInterp, motorDiary, ...
    medTime, vecDyskOnset, vecDyskOffset, adaptiveMetaData, ...
    strSide, cfg, varargin)
% powData - single channel data
% LDData - single channel data

%% Input parsing

% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'boolPlotSmooth', true, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'lenSmoothSec', 20, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'origAlpha', 0.3, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'smoothAlpha', 1, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

addParameter(p, 'LD1time', nan);
addParameter(p, 'LD1data', nan);

% color arguments
addParameter(p, 'color', [0, 0.4470, 0.7410], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'smoothColor', [1, 0, 0], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

% plotting limits
addParameter(p, 'ylimPow', [0, 1500], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimLD', [0, 500], ...
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
addParameter(p, 'ylimPKGDysk', [-5, 150], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimMDRange', [-0.5, 3.5], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

% output parameters
addParameter(p, 'boolReturnOutput', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'outputStruct', NaN);

% compat from outer
addParameter(p, 'boolSaveAsFig', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
boolPlotSmooth = p.Results.boolPlotSmooth;
lenSmoothSec = p.Results.lenSmoothSec;
origAlpha = p.Results.origAlpha;
smoothAlpha = p.Results.smoothAlpha;

LD1time = p.Results.LD1time;
LD1data = p.Results.LD1data;

color = p.Results.color;
smoothColor = p.Results.smoothColor;

ylimPow = p.Results.ylimPow;
ylimLD = p.Results.ylimLD;
ylimState = p.Results.ylimState;
ylimCurrent = p.Results.ylimCurrent;
ylimStateTrans = p.Results.ylimStateTrans;
ylimAWProb = p.Results.ylimAWProb;
ylimPKGBrady = p.Results.ylimPKGBrady;
ylimPKGDysk = p.Results.ylimPKGDysk;
ylimMDRange = p.Results.ylimMDRange;

boolReturnOutput = p.Results.boolReturnOutput;
outputStruct = p.Results.outputStruct;

boolSaveAsFig = p.Results.boolSaveAsFig;

% define any magic numbers
TIME_RES_MD_IN_MINUTES = 30;

%% parsing

% parse from metadata
strLD = adaptiveMetaData.strLD; 
strCh = adaptiveMetaData.strCh{1}; 
strPB = adaptiveMetaData.strPB{1};
boolActiveLD = adaptiveMetaData.boolActive;

% obtain the side of the recording
if strcmp(cfg.str_sub, 'RCS02')
    strMDBradyLabel = 'MD Brady Intensity';
    strMDDyskLabel = 'MD Dysk Intensity';
    strAWDispLabel = 'AW Dysk Prob';

    if strcmp(strSide, 'Left')
        strMDBrady = 'RSlowness';
        strMDBradyTbsome = 'RSlownessTbsome';
        
        strMDDysk = 'LDysk';
        strMDDyskTbsome = 'LDyskTbsome';
        
    elseif strcmp(strSide, 'Right')
        strMDBrady = 'LSlowness';
        strMDBradyTbsome = 'LSlownessTbsome';

        strMDDysk = 'LDysk';
        strMDDyskTbsome = 'LDyskTbsome';
    else
        error('Unknown side');
    end

elseif strcmp(cfg.str_sub, 'RCS14')
    ylimAWProb = [-1, 6];
    strMDBradyLabel = 'MD Brady Intensity';
    strMDDyskLabel = 'MD Trem Intensity';
    strAWDispLabel = 'AW Acceleration';
    strMDJawTremorLabel = 'MD Jaw Tremor Intsy';

    strMDBrady = 'RHandSlowness';
    strMDBradyTbsome = 'RHandSlownessTbsome';
    
    strMDDysk = 'RHandTremor';
    strMDDyskTbsome = 'RHandTremorTbsome';

    strMDJawTremor = 'JawTremor';

    ylimPKGDysk = [-5, 40];

elseif strcmp(cfg.str_sub, 'RCS17')
    strMDBradyLabel = 'MD Brady Intensity';
    strMDDyskLabel = 'MD Dysk Intensity';
    strAWDispLabel = 'AW Dysk Prob';

    if strcmp(strSide, 'Left')
        strMDBrady = 'RBodyBrady';
        strMDBradyTbsome = 'RBodyBradyTbsome';
        
        strMDDysk = 'RBodyDysk';
        strMDDyskTbsome = 'RBodyDyskTbsome';
        
    elseif strcmp(strSide, 'Right')
        strMDBrady = 'LBodyBrady';
        strMDBradyTbsome = 'LBodyBradyTbsome';

        strMDDysk = 'LBodyDysk';
        strMDDyskTbsome = 'LBodyDyskTbsome';
    else
        error('Unknown side');
    end

else
    error('Unknown subject')
end

% special case for one day of motor diary data
if strcmp(cfg.str_data_day, '20221024')
    ylimMDRange = [-0.5, 5.5];
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
boolMDValid = ~any(strcmp(cfg.str_no_md_data_day, cfg.str_data_day));

subplotNum = 3 + double(boolAWValid) + ...
    double(boolPKGValid) + double(boolMDValid);

if subplotNum == 6  % all three exists
    figTD_LD = figure('position', [10, 10, 2200, 1500]);
elseif subplotNum == 5  % just two exists
    figTD_LD = figure('position', [10, 10, 2200, 1500]);
elseif subplotNum == 4
    figTD_LD = figure('position', [10, 10, 2200, 1400]);
else
    figTD_LD = figure('position', [10, 10, 2200, 1200]);
end

idxCurrSubplot = 1;

% %% plotting the raw power data
% 
% vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
% 
% if ~boolPlotSmooth
%     plot(timePow, powData, 'Color', color, 'HandleVifsibility','off');
% else
%     % plot the original unsmoothed data
%     plot(timePow, powData, 'Color', [color, origAlpha], ...
%         'DisplayName', 'Unsmoothed');
% 
%     % now compute the moving average
%     fsPowCurr = seconds(mode(diff(timePow)));
%     powDataSmooth = movmean(powData, round(1/fsPowCurr * lenSmoothSec));
%     plot(timePow, powDataSmooth, 'Color', [smoothColor, smoothAlpha], ...
%         'LineWidth', 0.5, 'DisplayName', 'Smoothed')
% 
%     legend('boxoff');
% end
% 
% % plot times for when med was taken
% for i = 1:numel(medTime)
%     xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
% end
% 
% ylim(ylimPow);
% ylabel('Power (A.U.)')
% title(sprintf("Power band streamed: %s", strcat(strCh,' | ', strPB)))
% idxCurrSubplot = idxCurrSubplot + 1;


%% plot the LD data

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;

if isnan(LD1data)
    if ~boolPlotSmooth
        % no smoothing is performed
        plot(timeLD, LDData, 'Color', color, 'HandleVisibility','off');
    else
        % plot the original data
        plot(timeLD, LDData, 'Color', [color, origAlpha], ...
            'HandleVisibility','off');
        
        % plot the smoothed data
        if numel(LDData) > 1
            % now compute the moving average
            fsLDCurr = seconds(mode(diff(timeLD)));
            LDDataSmooth = movmean(LDData, round(1/fsLDCurr * lenSmoothSec));
            plot(timeLD, LDDataSmooth, 'Color', [smoothColor, smoothAlpha], ...
                'LineWidth', 0.5, 'HandleVisibility','off')
        end
    end
    
    xlimLD = [timeLD(1), timeLD(end)];
    ylim(ylimLD);
else
    % plot the original data
    yyaxis left;
%     plot(timeLD, LDData, 'Color', [color, origAlpha], ...
%         'HandleVisibility','off');
    
    % plot the smoothed data
    if numel(LDData) > 1
        % now compute the moving average
        fsLDCurr = seconds(mode(diff(timeLD)));
        LDDataSmooth = movmean(LDData, round(1/fsLDCurr * lenSmoothSec));
        plot(timeLD, LDDataSmooth, 'Color', [color, smoothAlpha], ...
            'LineWidth', 0.5, 'HandleVisibility','off')
    end
    ylim(ylimLD);

    % also plot the LD1 data
    yyaxis right;
%     plot(LD1time, LD1data, 'Color', [0.3010, 0.7450, 0.9330, origAlpha], ...
%         'HandleVisibility','off');

    LD1DataSmooth = movmean(LD1data, round(1/fsLDCurr * lenSmoothSec));
    plot(LD1time, LD1DataSmooth, 'Color', [0.8500, 0.3250, 0.0980, smoothAlpha], ...
        'LineWidth', 0.5, 'HandleVisibility','off')
    ylim([0, 40]);

    xlimLD = [timeLD(1), timeLD(end)];
end

% plot times for when med was taken
if boolMDValid
    for i = 1:numel(medTime)
        xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
    end
end

% add current LD threshold
if strcmp(strLD, 'LD0')
    strColorLDThresh = 'g';
elseif strcmp(strLD, 'LD1')
    strColorLDThresh = 'k';
end

yline(adaptiveMetaData.thresh(1), '--', 'Color', strColorLDThresh, 'LineWidth', 2.5, ...
    'DisplayName', sprintf('%s threshold', strLD));
if adaptiveMetaData.boolDualThresh
    yline(adaptiveMetaData.thresh(end), '--', 'Color', strColorLDThresh, 'LineWidth', 2.5, ...
        'DisplayName', sprintf('%s threshold', strLD));
end

% change tick size
ax = gca;
ax.FontSize = tickFontSize; 

% add label
ylabel("LD Value (A.U.)", 'FontSize', labelFontSize);
% ylim(ylimLD);

% add title
if ~adaptiveMetaData.boolDualThresh
    strThresh = sprintf('Thresh %d', adaptiveMetaData.thresh(1));
else
    strThresh = sprintf('Thresh %d-%d', adaptiveMetaData.thresh(1), ...
        adaptiveMetaData.thresh(2));
end

if ~boolActiveLD
    strActive = 'LD non-active';
else
    strActive = 'LD Active';
end
title(sprintf("%s %s Streamed: %s, %s, %s, %s, %s, %s", strSide, strLD, ...
    strcat(strCh,' | ', strPB), ...
    strThresh, sprintf('UR %.1fs', adaptiveMetaData.updateRate), ...
    sprintf('O/T DUR %ds/%ds', adaptiveMetaData.onsetDuration, adaptiveMetaData.terminationDuration), ...
    sprintf('Shift %d', adaptiveMetaData.idxShift), strActive), ...
    'FontSize', titleFontSize)

idxCurrSubplot = idxCurrSubplot + 1;


%%  plot the states

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
if ~boolPlotSmooth
    % plot the unsmoothed data
    plot(timeState, stateData, 'Color', color)
else
    % plot the original data
    plot(timeState, stateData, 'Color', [color, origAlpha]);

    % now compute the moving average
    if numel(stateData) > 1
        fsAdapCurr = seconds(mode(diff(timeState)));
        stateDataSmooth = movmean(stateData, round(1/fsAdapCurr * lenSmoothSec));
        plot(timeState, stateDataSmooth, 'Color', [smoothColor, smoothAlpha], ...
            'LineWidth', 0.5, 'DisplayName', 'Smoothed')
    end
end

% plot times for when med was taken
if boolMDValid
    for i = 1:numel(medTime)
        xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
    end
end

% change tick size
ax = gca;
ax.FontSize = tickFontSize; 

% add label
ylabel("Decoder State", 'FontSize', labelFontSize)
ylim(ylimState)

% add title
if adaptiveMetaData.boolActive
    title('Decoder State Streamed', 'FontSize', titleFontSize)
else
    title(sprintf('Decoder State if Driven by %s', strLD) ,...
        'FontSize', titleFontSize)
end

idxCurrSubplot = idxCurrSubplot + 1;

%% plot the current injected

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
if ~boolPlotSmooth
    % plot the unsmoothed data
    plot(timeState, currentData, 'Color', color)
else
    % plot the original data
    plot(timeState, currentData, 'Color', [color, origAlpha]);
    
    if numel(currentData) > 1
        % now compute the moving average
        currentDataSmooth = movmean(currentData, round(1/fsAdapCurr * lenSmoothSec));
        plot(timeState, currentDataSmooth, 'Color', [smoothColor, smoothAlpha], ...
            'LineWidth', 0.5, 'DisplayName', 'Smoothed')
    end
end

% plot times for when med was taken
if boolMDValid
    for i = 1:numel(medTime)
        xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
    end
end

% change tick size
ax = gca;
ax.FontSize = tickFontSize; 

% add label
ylabel("Current Injected (mA)", 'FontSize', labelFontSize)
ylim(ylimCurrent)

% add title
if adaptiveMetaData.boolActive
    title('Current Injected Streamed', 'FontSize', titleFontSize)
else
    title(sprintf('Current Injected if Driven by %s', strLD), ...
        'FontSize', titleFontSize)
end

idxCurrSubplot = idxCurrSubplot + 1;

% %% estimate the average speed of state transition
% 
% % compute sample diff and figure out rate of change
% if numel(stateData) > 1
%     vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
% 
%     stateDataDiff = diff(stateData);
%     idxStateFull = 1:numel(stateDataDiff);
%     
%     if ~boolPlotSmooth
%         fsAdapCurr = seconds(mode(diff(timeState)));
%     end
%     
%     idxChange = idxStateFull(stateDataDiff ~=0);
%     if numel(idxChange) ~= 0
%         stateChangeRate = diff([1, idxChange]) * fsAdapCurr;
%         timeChangeRate = timeState(idxChange);
%     else
%         timeChangeRate = timeState;
%         stateChangeRate = zeros(size(timeChangeRate));
%     end
%     % now generate the plot
%     plot(timeChangeRate, stateChangeRate)
%     for i = 1:numel(medTime)
%         xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
%     end
%     
%     % change tick size
%     ax = gca;
%     ax.FontSize = tickFontSize; 
%     
%     % add label
%     ylabel("Time (s)", 'FontSize', labelFontSize);
%     ylim(ylimStateTrans);
%     
%     % add title
%     title("Time between State Transition", 'FontSize', titleFontSize)
%     
%     idxCurrSubplot = idxCurrSubplot + 1;
% 
% end

%% now plot the apple watch data if it exists

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
    xlim(xlimLD);
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
    plot(timePKG, PKGDyskData, 'LineWidth', 2, ...
        'DisplayName', 'PKG Dysk Score');
    yline(10, 'Color', [0.8500, 0.3250, 0.0980], 'LineStyle', '--', ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');
    ax = gca;
    ax.FontSize = tickFontSize;
    ylabel("Dysk score (a.u.)", 'FontSize', labelFontSize)
    ylim(ylimPKGDysk)
    
    xlim(xlimLD);
%     % plot the tremor scores
%     yyaxis right;
%     plot(timePKG, PKGDyskData, 'LineWidth', 2, ...
%         'DisplayName', 'PKG Trem Score');
% 
%     ax = gca;
%     ax.FontSize = tickFontSize;
%     ylabel("Trem score (a.u.)", 'FontSize', labelFontSize)
%     ylim(ylimPKGDysk)

    title("PKG data", 'FontSize', titleFontSize)
    lgdCurr = legend('boxoff');
    lgdCurr.FontSize = legendFontSize;
    idxCurrSubplot = idxCurrSubplot + 1;
end

%% now plot the motor diary dysk


if ~any(strcmp(cfg.str_no_md_data_day, cfg.str_data_day))
    vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
    
    % plot the brady and dysk intensities
    plot(motorDiaryInterp.time, motorDiaryInterp.(strMDBrady), ...
        "DisplayName", strMDBradyLabel, 'LineWidth', 2);
    plot(motorDiaryInterp.time, motorDiaryInterp.(strMDDysk), ...
        "DisplayName", strMDDyskLabel, 'LineWidth', 2);
    
    if strcmp(cfg.str_sub, 'RCS14')
        plot(motorDiaryInterp.time, motorDiaryInterp.(strMDJawTremor), ...
            "DisplayName", strMDJawTremorLabel, 'LineWidth', 2);
    end
    
    % plot times for when med was taken
    for i = 1:numel(medTime)
        xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
    end
    
    % change tick size
    ax = gca;   
    ax.FontSize = tickFontSize;
    
    % add label
    ylabel('MD Intensity', 'FontSize', labelFontSize)
    xlim(xlimLD);
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
end

%% saving output figure

% change the x axies
linkaxes([vecAxes{:}], 'x');
datetick('x','HH:MM');

% save the output figure
fFigure = sprintf('%s_%s_%s_Fluc_%s_%s', cfg.str_sub, ...
    cfg.str_aDBS_paradigm(1:5), strLD, cfg.str_data_day, ...
    cfg.vec_str_side{1}(1));

%% specify the output struct and storing all useful information

if boolReturnOutput
    % compute the total amount of time spent in on and off
    perMD_ON = sum(motorDiary.ON) / 24;
    perMD_OFF = sum(motorDiary.OFF) / 24;

    % compute the total amount of time spent in brady and dysk
    perMDBrady = sum(motorDiary.(strMDBrady) > 0) / 24;
    perMDBradyTbsome = sum(motorDiary.(strMDBradyTbsome) > 0) / 24;
    perMDDysk = sum(motorDiary.(strMDDysk) > 0) / 24;
    perMDDyskTbsome = sum(motorDiary.(strMDDyskTbsome) > 0) / 24;
    
    % concatenate all non-zero brady and dysk intensity
    idxMDBrady = motorDiary.(strMDBrady) > 0;
    if sum(idxMDBrady) > 0
        allMDBrady = motorDiary.(strMDBrady)(idxMDBrady);
    else
        allMDBrady = [];
    end

    idxMDDysk = motorDiary.(strMDDysk) > 0;
    if sum(idxMDDysk) > 0
        allMDDysk = motorDiary.(strMDDysk)(idxMDDysk);
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