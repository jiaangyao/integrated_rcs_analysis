function plotTDFlucOverTimeCxtWin(timeResCxtWin, ...
    timePow, powData, timeLD, LDData, ...
    timeState, stateData, currentData, ... % timeAWTre, AWTremData, ...
    timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
    motorDiaryInterp, motorDiary, ...
    medTime, vecDyskOnset, vecDyskOffset, adaptiveMetaData, ...
    strSide, pFigure, cfg, varargin)
% functional wrapper for plotting figures in continous windows of time

%% Input parsing

% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'timeRelaxBound', 1e-3, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'boolSaveAsFig', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% parse input (only part of the flag)
parse(p,varargin{:});
timeRelaxBound = p.Results.timeRelaxBound;
boolSaveAsFig = p.Results.boolSaveAsFig;

% define any magic number
SECONDS_IN_HR = 3600;

%% function defn

% first attempt to figure our how many hours are data are contained in the
% recording
vecUniqueHrs = unique(hour(timePow));
initDateTime = timePow(1);
initDateTime.Minute = 0;
initDateTime.Second = 0;

% sanity check
if ~(strcmp(initDateTime.Format, timePow(end).Format) && ...
        strcmp(initDateTime.TimeZone, timePow(end).TimeZone))
    error('Initial time should have same meta')
end

% define the time ticks
vecHrTicks = [initDateTime];
for i = 1:(floor(numel(vecUniqueHrs) / timeResCxtWin) + 1)
    vecHrTicks = [vecHrTicks, vecHrTicks(end) + ...
        seconds(round(timeResCxtWin * SECONDS_IN_HR))];
end

% now loop through the different hours
for idxHr = 1:(numel(vecHrTicks) -1)
    startTimeCurr = vecHrTicks(idxHr);
    endTimeCurr = vecHrTicks(idxHr + 1);

    idxPowHrCurr = timePow >= vecHrTicks(idxHr) & ...
        timePow < vecHrTicks(idxHr + 1);
    timePowCurr = timePow(idxPowHrCurr);
    powDataCurr = powData(idxPowHrCurr);

    idxLDHrCurr = timeLD >= vecHrTicks(idxHr) & ...
        timeLD < vecHrTicks(idxHr + 1);
    timeLDCurr = timeLD(idxLDHrCurr);
    LDDataCurr = LDData(idxLDHrCurr);
    
    idxStateHrCurr = timeState >= vecHrTicks(idxHr) & ...
        timeState < vecHrTicks(idxHr + 1);
    timeStateCurr = timeState(idxStateHrCurr);
    stateDataCurr = stateData(idxStateHrCurr);
    currentDataCurr = currentData(idxStateHrCurr);
    
    % optionally process AW data
    if ~any(strcmp(cfg.str_no_aw_data_day, cfg.str_data_day))
        idxAWHrCurr = timeAWDysk >= vecHrTicks(idxHr) & ...
            timeAWDysk < vecHrTicks(idxHr + 1);
        timeAWDyskCurr = timeAWDysk(idxAWHrCurr);
        AWDyskDataCurr = AWDyskData(idxAWHrCurr);
    else
        timeAWDyskCurr = NaN; AWDyskDataCurr= NaN;
    end

    % optionally process PKG data
    if ~any(strcmp(cfg.str_no_pkg_data_day, cfg.str_data_day))
        idxPKGHrCurr = timePKG >= vecHrTicks(idxHr) & ...
            timePKG < vecHrTicks(idxHr + 1);
        timePKGCurr = timePKG(idxPKGHrCurr);
        PKGBradyDataCurr = PKGBradyData(idxPKGHrCurr);
        PKGDyskDataCurr = PKGDyskData(idxPKGHrCurr);
    else
        timePKGCurr = NaN; PKGBradyDataCurr = NaN; PKGDyskDataCurr = NaN;
    end

    medTimeCurr = [];
    for i = 1:numel(medTime)
        if medTime(i) >= vecHrTicks(idxHr) & ...
            medTime(i) < vecHrTicks(idxHr + 1)
            medTimeCurr = [medTimeCurr, medTime(i)];
        end
    end
    
    % also extract the yline stuff
    vecDyskOnsetCurr = [];
    for i = 1:numel(vecDyskOnset)
        if vecDyskOnset(i) >= vecHrTicks(idxHr) & ...
            vecDyskOnset(i) < vecHrTicks(idxHr + 1)
            vecDyskOnsetCurr = [vecDyskOnsetCurr, vecDyskOnset(i)];
        end
    end

    vecDyskOffsetCurr = [];
    for i = 1:numel(vecDyskOffset)
        if vecDyskOffset(i) >= vecHrTicks(idxHr) & ...
            vecDyskOffset(i) < vecHrTicks(idxHr + 1)
            vecDyskOffsetCurr = [vecDyskOffsetCurr, vecDyskOffset(i)];
        end
    end
    
    try
        [figTDCurr, fFigureHandleCurr, ~] = ...
            plotTDFluc(timePowCurr, powDataCurr, timeLDCurr, LDDataCurr, ...
                timeStateCurr, stateDataCurr, currentDataCurr, ...
                timeAWDyskCurr, AWDyskDataCurr, timePKGCurr, PKGBradyDataCurr, ...
                PKGDyskDataCurr, motorDiaryInterp, motorDiary, ...
                medTimeCurr, vecDyskOnsetCurr, vecDyskOffsetCurr, ...
                adaptiveMetaData, strSide, cfg, ...
                varargin{:});
    catch
        close all;
        warning('Plotting failed for time window %d - %d', ...
            startTimeCurr.Hour, endTimeCurr.Hour)
        continue;
    end
    
    plotRelaxWin = minutes(round(timeRelaxBound * timeResCxtWin * ...
        SECONDS_IN_HR));
    startTimePlotCurr = startTimeCurr - plotRelaxWin;
    endTimePlotCurr = endTimeCurr + plotRelaxWin;
    xlim([startTimePlotCurr, endTimePlotCurr]);
    
    % now try to save the figure
    fFigureCurr = sprintf('%s_%d_%d', fFigureHandleCurr, ...
        hour(startTimeCurr), hour(endTimeCurr));

    % create the output folder
    pFigureCurr = fullfile(pFigure, sprintf('%s_CxtWin_%d', ...
        adaptiveMetaData.strLD, timeResCxtWin));
    if ~exist(pFigureCurr, 'dir')
        mkdir(pFigureCurr)
    end
    
    % save figure
    if boolSaveAsFig
        savefig(figTDCurr, fullfile(pFigureCurr, fFigureCurr))
        close(figTDCurr);
    else
        saveas(figTDCurr, fullfile(pFigureCurr, sprintf('%s.png',fFigureCurr)))
        close(figTDCurr);
    end

end