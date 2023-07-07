function [figLogs, fFigure] = plotLogswPKG(time, state, amp, aliasedState, timePKG, PKGBradyData, ...
    PKGDyskData, adaptiveMetaData, strSide, cfg, varargin)
%% Input parsing

% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'boolPlotSmooth', true, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'lenSmoothSec', 30, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'origAlpha', 0.3, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'smoothAlpha', 1, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

% color arguments
addParameter(p, 'color', [0, 0.4470, 0.7410], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'smoothColor', [1, 0, 0], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

% plotting limits
addParameter(p, 'ylimState', [-0.5, 4.5], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimAliasedState', [-0.5, 1.5], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimCurrent', [0.5, 4], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimPKGBrady', [-5, 70], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'ylimPKGDysk', [-5, 150], ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

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

color = p.Results.color;
smoothColor = p.Results.smoothColor;

ylimState = p.Results.ylimState;
ylimAliasedState = p.Results.ylimAliasedState;
ylimCurrent = p.Results.ylimCurrent;
ylimPKGBrady = p.Results.ylimPKGBrady;
ylimPKGDysk = p.Results.ylimPKGDysk;

boolSaveAsFig = p.Results.boolSaveAsFig;

%% parsing

% parse from metadata
strLD = adaptiveMetaData.strLD; 
strCh = adaptiveMetaData.strCh{1}; 
strPB = adaptiveMetaData.strPB{1};
boolActiveLD = adaptiveMetaData.boolActive;

% set various font sizes
titleFontSize = 16;
labelFontSize = 16;
legendFontSize = 13;
tickFontSize = 13;

% set the number of figures and figure size
subplotNum = 4;
figLogs = figure('position', [10, 10, 2200, 1500]);
idxCurrSubplot = 1;

%% Plotting the state data

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
if ~boolPlotSmooth
    % no smoothing is performed
    plot(time, state, 'Color', color, 'HandleVisibility','off');
else
    % plot the original data
    plot(time, state, 'Color', [color, origAlpha], ...
        'HandleVisibility','off');

    % plot the smoothed data
    if numel(state) > 1
        % now compute the moving average
        fsStateCurr = seconds(mode(diff(time)));
        stateDataSmooth = movmean(state, round(1/fsStateCurr * lenSmoothSec));
        plot(time, stateDataSmooth, 'Color', [smoothColor, smoothAlpha], ...
            'LineWidth', 0.5, 'HandleVisibility','off')
    end
end

% change tick size
ax = gca;
ax.FontSize = tickFontSize; 

% add label
ylabel("State Value", 'FontSize', labelFontSize);
ylim(ylimState);

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
title(sprintf("%s %s States: %s, %s, %s, %s, %s, %s", strSide, strLD, ...
    strcat(strCh,' | ', strPB), ...
    strThresh, sprintf('UR %.1fs', adaptiveMetaData.updateRate), ...
    sprintf('O/T DUR %d/%d', adaptiveMetaData.onsetDuration, adaptiveMetaData.terminationDuration), ...
    sprintf('Shift %d', adaptiveMetaData.idxShift), strActive), ...
    'FontSize', titleFontSize)

idxCurrSubplot = idxCurrSubplot + 1;


%% Plotting the aliased state data

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
if ~boolPlotSmooth
    % no smoothing is performed
    plot(time, aliasedState, 'Color', color, 'HandleVisibility','off');
else
    % plot the original data
    plot(time, aliasedState, 'Color', [color, origAlpha], ...
        'HandleVisibility','off');

    % plot the smoothed data
    if numel(aliasedState) > 1
        % now compute the moving average
        aliasedStateDataSmooth = movmean(aliasedState, round(1/fsStateCurr * lenSmoothSec));
        plot(time, aliasedStateDataSmooth, 'Color', [smoothColor, smoothAlpha], ...
            'LineWidth', 0.5, 'HandleVisibility','off')
    end
end

% change tick size
ax = gca;
ax.FontSize = tickFontSize; 

% add label
ylabel("Aliased State Value", 'FontSize', labelFontSize);
ylim(ylimAliasedState);

% add title
title("Aliased States Formed by Merging States with Same Stim")

idxCurrSubplot = idxCurrSubplot + 1;


%% Plotting the amp data

vecAxes{idxCurrSubplot} = subplot(subplotNum, 1, idxCurrSubplot); hold on;
if ~boolPlotSmooth
    % no smoothing is performed
    plot(time, amp, 'Color', color, 'HandleVisibility','off');
else
    % plot the original data
    plot(time, amp, 'Color', [color, origAlpha], ...
        'HandleVisibility','off');

    % plot the smoothed data
    if numel(amp) > 1
        ampDataSmooth = movmean(amp, round(1/fsStateCurr * lenSmoothSec));
        plot(time, ampDataSmooth, 'Color', [smoothColor, smoothAlpha], ...
            'LineWidth', 0.5, 'HandleVisibility','off')
    end
end

% change tick size
ax = gca;
ax.FontSize = tickFontSize; 

% add label
ylabel("Current Injected (mA)", 'FontSize', labelFontSize)
ylim(ylimCurrent)

% add title
title('Current Injected Streamed', 'FontSize', titleFontSize)

idxCurrSubplot = idxCurrSubplot + 1;

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

    title("PKG data", 'FontSize', titleFontSize)
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
