function [figBox, fFigure, outputStruct] = plotBoxPlot(timeState, stateData, ...
    motorDiary, vecDyskOnset, vecDyskOffset, adaptiveMetaData, cfgIn, varargin)
%% Input parsing
% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'boolStreamed', true, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));

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
boolStreamed = p.Results.boolStreamed;

boolReturnOutput = p.Results.boolReturnOutput;
outputStruct = p.Results.outputStruct;

% define any magic numbers
TIME_RES_MD_IN_MINUTES = 30;

COLOR_HIGH_STIM = [0.4660 0.6740 0.1880];
COLOR_LOW_STIM = [0.6350 0.0780 0.1840];

%% parsing input

if adaptiveMetaData.boolReverse
    highState = adaptiveMetaData.statesFull(1);
    lowState = adaptiveMetaData.statesFull(end);
else
    highState = adaptiveMetaData.statesFull(end);
    lowState = adaptiveMetaData.statesFull(1);
end

strHighState = sprintf('High Stim (State %d)', highState);
strLowState = sprintf('Low Stim (State %d)', lowState);

colorHighState = COLOR_HIGH_STIM;
colorLowState = COLOR_LOW_STIM;

% make a local copy of cfg
cfg = cfgIn;

% set various font sizes
titleFontSize = 16;
labelFontSize = 16;
legendFontSize = 13;
tickFontSize = 13;

%% Parse the OFF states

% first compute the on and off redacted tables
idxON = (motorDiary.ON == 1);
onRedacted = motorDiary(idxON, :);

idxOFF = (motorDiary.OFF == 1);
offRedacted = motorDiary(idxOFF, :);

% correct onsets and offsets to have the same length
if numel(vecDyskOnset) ~= numel(vecDyskOffset) && numel(vecDyskOnset) > 1
    [vecDyskOnset, vecDyskOffset] = ...
        dtCorrectMarkerStartEnd(vecDyskOnset, vecDyskOffset);
end

% append corrected onset and offsets to output struct if returning
if boolReturnOutput
    totalDyskTime = 0;
    if ~isempty(vecDyskOnset) && ~isempty(vecDyskOffset)
        for i = 1:numel(vecDyskOnset)
            totalDyskTime = totalDyskTime + ...
                minutes(vecDyskOffset(i) - vecDyskOnset(i));
        end
    end
    
    outputStruct.vecDyskOnset = vecDyskOnset;
    outputStruct.vecDyskOffset = vecDyskOffset;
    outputStruct.totalDyskTime = totalDyskTime;
end

% now estimate the total proportion of stim in each state
timeStateOFF_Full = [];
stateOFF_Full = [];
for i = 1:size(offRedacted, 1)
    % obtain the relevant indices
    endTime = offRedacted.time(i);
    startTime = endTime - minutes(TIME_RES_MD_IN_MINUTES);
    idxCurrOff = timeState >= startTime & timeState <= endTime;
    
    % append to outer list
    timeStateOFF_Full = [timeStateOFF_Full; timeState(idxCurrOff)];
    stateOFF_Full = [stateOFF_Full; stateData(idxCurrOff)];
end

%% Parse the ON states

% if we have enough patient-reported dysk labels to parse data with
if min([numel(vecDyskOnset), numel(vecDyskOffset)]) >= 1
    % now estimate for the various ON states as well
    stateON_Full = [];
    stateONGood_Full = [];
    stateONDysk_Full = [];

    timeStateON_Full = [];
    timeStateONGood_Full = [];
    timeStateONDysk_Full = [];
    for i = 1:size(onRedacted, 1)
        endTime = onRedacted.time(i);
        startTime = endTime - minutes(TIME_RES_MD_IN_MINUTES);

        idxCurrON = timeState >= startTime & timeState <= endTime;
        timeStateCurr = timeState(idxCurrON);
        stateDataCurr = stateData(idxCurrON);

        timeStateON_Full = [timeStateON_Full; timeStateCurr];
        stateON_Full = [stateON_Full; stateDataCurr];

        % loop through all Dysk time pairs and see if any time falls in the
        % range
        idxDyskFull = [];
        for j = 1:numel(vecDyskOnset)
            idxDyskCurr = timeStateCurr >= vecDyskOnset(j) & ...
                timeStateCurr < vecDyskOffset(j);
            idxDyskFull = [idxDyskFull; idxDyskCurr];

            % if any are in the current dyskinesia range
            if sum(idxDyskCurr) > 0
                stateONDysk_Full = [stateONDysk_Full; ...
                    stateDataCurr(idxDyskCurr)];
                stateONGood_Full = [stateONGood_Full; ...
                    stateDataCurr(~idxDyskCurr)];

                timeStateONDysk_Full = [timeStateONDysk_Full; ...
                    timeStateCurr(idxDyskCurr)];
                timeStateONGood_Full = [timeStateONGood_Full; ...
                    timeStateCurr(~idxDyskCurr)];
            end
        end

        if sum(idxDyskFull) == 0
            stateONGood_Full = [stateONGood_Full; stateDataCurr];
            timeStateONGood_Full = [timeStateONGood_Full; timeStateCurr];
        end
    end


    % sanity check
    if numel(timeStateONGood_Full) ~= numel(stateONGood_Full) || ...
            numel(timeStateONDysk_Full) ~= numel(stateONDysk_Full)
        error('These should have the same number of elements')
    end

    timeONCombined = sort([timeStateONDysk_Full; timeStateONGood_Full]);
    if ~all(timeONCombined == timeStateON_Full)
        error("These two times should be equal")
    end

% otherwise no parsing
else
    stateON_Full = [];
    timeStateON_Full = [];

    for i = 1:size(onRedacted, 1)
        endTime = onRedacted.time(i);
        startTime = endTime - minutes(TIME_RES_MD_IN_MINUTES);

        idxCurrON = timeState >= startTime & timeState <= endTime;
        timeStateCurr = timeState(idxCurrON);
        stateDataCurr = stateData(idxCurrON);

        timeStateON_Full = [timeStateON_Full; timeStateCurr];
        stateON_Full = [stateON_Full; stateDataCurr];
    end

end

if numel(timeStateOFF_Full) ~= numel(stateOFF_Full) || ...
        numel(timeStateON_Full) ~= numel(stateON_Full)
    error('These should have the same number of elements')
end

% additional sanity check
if (sum(idxON) + sum(idxOFF)) == size(motorDiary, 1)
    timeONOFFCombined = sort([timeStateOFF_Full; timeStateON_Full]);
    idxTimeStateRedact = timeState >= ...
        (motorDiary.time(1) - minutes(TIME_RES_MD_IN_MINUTES)) & ...
        timeState < motorDiary.time(end);
    timeStateRedact = timeState(idxTimeStateRedact);

    if ~all(timeONOFFCombined == timeStateRedact)
        error("These two should be equal")
    end
end

%% now compute the proportions

% first compute for normal ON and OFF states
if size(offRedacted, 1) >= 1
    perOFFStateLow = sum(stateOFF_Full == lowState) / size(stateOFF_Full, 1);
    perOFFStateHigh = sum(stateOFF_Full == highState) / size(stateOFF_Full, 1);
end

perONStateLow = sum(stateON_Full == lowState) / size(stateON_Full, 1);
perONStateHigh = sum(stateON_Full == highState) / size(stateON_Full, 1);

% compute separately for dysk times and good ON state if patient labels
% exist
if min([numel(vecDyskOnset), numel(vecDyskOffset)]) >= 1
    perONStateGoodLow = sum(stateONGood_Full == lowState) / ...
        size(stateONGood_Full, 1);
    perONStateGoodHigh = sum(stateONGood_Full == highState) / ...
        size(stateONGood_Full, 1);
    
    perONStateDyskLow = sum(stateONDysk_Full == lowState) / ...
        size(stateONDysk_Full, 1);
    perONStateDyskHigh = sum(stateONDysk_Full == highState) / ...
        size(stateONDysk_Full, 1);
end

% also compute total proportion
perFullStateLow = sum(stateData == lowState) / size(stateData, 1);
perFullStateHigh = sum(stateData == highState) / size(stateData, 1);

%% organize the variables for plotting

% note order here matters
if min([numel(vecDyskOnset), numel(vecDyskOffset)]) >= 1
    if size(offRedacted, 1) >= 1
        matrixPerFull = [perFullStateLow, perFullStateHigh;
            perOFFStateLow, perOFFStateHigh;
            perONStateLow, perONStateHigh;
            perONStateGoodLow, perONStateGoodHigh;
            perONStateDyskLow, perONStateDyskHigh];
        x = categorical({'All', 'Off', 'On', 'On w/o Dysk', 'Dysk Only'}, ...
            {'All', 'Off', 'On', 'On w/o Dysk', 'Dysk Only'});
    else
        matrixPerFull = [perFullStateLow, perFullStateHigh;
            perONStateLow, perONStateHigh;
            perONStateGoodLow, perONStateGoodHigh;
            perONStateDyskLow, perONStateDyskHigh];
        x = categorical({'All', 'On', 'On w/o Dysk', 'Dysk Only'}, ...
            {'All', 'On', 'On w/o Dysk', 'Dysk Only'});
    end
    figBox = figure('position', [10, 10, 800, 1000]);
    
else
    if size(offRedacted, 1) >= 1
        matrixPerFull = [perFullStateLow, perFullStateHigh; ...
            perOFFStateLow, perOFFStateHigh; ...
            perONStateLow, perONStateHigh];
        x = categorical({'All', 'Off', 'On'}, {'All', 'Off', 'On'});
    else
        matrixPerFull = [perFullStateLow, perFullStateHigh;
            perONStateLow, perONStateHigh];
        x = categorical({'All', 'On'}, {'All','On'});
    end
    figBox = figure('position', [10, 10, 600, 1000]);

end


%% now plot the barplots

% generate the bar plot
bh = bar(x, matrixPerFull, 'stacked');

% change tick size
ax = gca;
ax.FontSize = tickFontSize; 

% now also change color
set(bh, 'FaceColor', 'Flat')
bh(1).CData = colorLowState;
bh(2).CData = colorHighState;

% change display name of the two labels
set(bh, {'DisplayName'}, {strLowState, strHighState}')

% add label to figure
ylabel('Proportion of time', 'FontSize', labelFontSize)
ylim([0, 1.1]);

% add title
if adaptiveMetaData.boolActive
    title(sprintf('Proportion of state data driven by %s', ...
        adaptiveMetaData.strLD), 'FontSize', titleFontSize);
else
    title(sprintf('Proportion of state data if driven by %s', ...
        adaptiveMetaData.strLD), 'FontSize', titleFontSize);
end

% add legend
lgdCurr = legend('boxoff');
lgdCurr.FontSize = legendFontSize;

set(gca,'box','off')

fFigure = sprintf('%s_%s_%s_Boxplot_%s_%s', cfg.str_sub, ...
    cfg.str_aDBS_paradigm(1:5), adaptiveMetaData.strLD, ...
    cfg.str_data_day, cfg.vec_str_side{1}(1));


end