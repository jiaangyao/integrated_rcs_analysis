function [figWearable, fFigure] = plotWearableComp(timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData,...
    motorDiaryInterp, motorDiary, ...
    medTime, vecDyskOnset, vecDyskOffset, cfg, varargin)
%% Input parsing
% adding custom input flag so can implement new path searching without
% modifying existing scripts - JY 07/18/2022

% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'boolStreamed', true, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));


parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

boolStreamed = p.Results.boolStreamed;

TIME_RES_MD_IN_MINUTES = 30;

%%
vecAxes = [];
figWearable = figure('position', [10, 10, 1600, 1200]); 

ax1 = subplot(411); hold on;

plot(timeAWDysk, AWDyskData, 'DisplayName', 'AW Dysk', ...
    'LineWidth', 2);

title("Apple Watch data")
ylim([0, 1])
ylabel("Probability")
legend('boxoff')
vecAxes = [vecAxes, ax1];

%%

ax2 = subplot(412); hold on;
plot(timePKG, PKGDyskData, 'LineWidth', 2, ...
    'DisplayName', 'Raw PKG Dysk Score');
ylim([-0.5, 20.1])
ylabel("Dysk score (a.u.)")
title('PKG Dysk Score Raw')
legend('boxoff')
vecAxes = [vecAxes, ax2];


%%

ax3 = subplot(413); hold on;
plot(timePKG, log10(PKGDyskData), 'LineWidth', 2, ...
    'DisplayName', 'Log PKG Dysk Score');
ylim([-1.1, 3.1])
ylabel("Log Dysk score (a.u.)")
title('PKG Dysk Log scaled')
legend('boxoff')
vecAxes = [vecAxes, ax3];

%%

ax4 = subplot(414); hold on; 
plot(motorDiaryInterp.time, motorDiaryInterp.LDysk, ...
    "DisplayName", 'MD Dysk Intsy');
for i = 1:numel(medTime)
    xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
end
ylabel('MD Intensity')
ylimMDRange = [-0.5, 4.5];
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

%% now fill in the motor diary ON/OFF times

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

title('Motor Diary Data')
legend('boxoff')

if size(motorDiaryInterp, 1) > 1
    vecAxes = [vecAxes, ax4];
end


linkaxes(vecAxes, 'x');
datetick('x','HH:MM');

t1 = 1;

%%

vecAxes = [];
figWearable = figure('position', [10, 10, 1600, 1200]); 

ax1 = subplot(211); hold on;
plot(timePKG, PKGBradyData, 'LineWidth', 2, ...
    'DisplayName', 'Raw PKG Brady Score');
ylim([-5, 155])
ylabel("Brady score (a.u.)")
title('PKG Brady Score Raw')
legend('boxoff')
vecAxes = [vecAxes, ax1];

%%

ax2 = subplot(212); hold on; 
plot(motorDiaryInterp.time, motorDiaryInterp.LSlowness, ...
    "DisplayName", 'MD Brady Intsy');
for i = 1:numel(medTime)
    xline(medTime(i), 'k--', 'LineWidth', 2, 'HandleVisibility','off')
end
ylabel('MD Intensity')
ylimMDRange = [-0.5, 4.5];
ylim(ylimMDRange)

% change the range of the plot
ylimRangeCurr = ylimMDRange;

%% now fill in the motor diary ON/OFF times

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

title('Motor Diary Data')
legend('boxoff')

if size(motorDiaryInterp, 1) > 1
    vecAxes = [vecAxes, ax4];
end


linkaxes(vecAxes, 'x');
datetick('x','HH:MM');

t1 = 1;

end

%%
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
    end

end
end