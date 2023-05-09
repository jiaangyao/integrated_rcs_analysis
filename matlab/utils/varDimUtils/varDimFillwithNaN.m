function [timewNaN, timeRelwNaN, datawNaN] = varDimFillwithNaN(time, ...
    timeRel, data, fs, TIME_COMPARISON_TOL)
%% varDimUtils script - fill  missing time points in data with NaNs
% Convert inputArray to tall array if it is wide
% 
% INPUT:
% time                  - datetime: time with missing time points potentially
% data                  - double: data with missing time points potentially
% fs                    - double: sampling rate
% TIME_COMPARISON_TOL   - double: tolerance for difference in time


%% fill data with NaN at inconsistencies

% first find where time differences are greater than sampling period
T = 1/fs;
idxSkip = find(abs(seconds(diff(time)) - T) > TIME_COMPARISON_TOL);

% loop through all inconsistencies and fill with nan accordingly
timewNaN = [];
timeRelwNaN = [];
datawNaN = [];
boolFsInconsistent = false; 
for i = 1:numel(idxSkip)
    % obtain current start and end indices
    if i == 1
        idxStart = 1;
    else
        idxStart = idxSkip(i - 1) + 1;
    end
    idxEnd = idxSkip(i);

    % first obtain the number of nan points needed
    timeElapsed = seconds(time(idxEnd + 1) - time(idxEnd));
    mod = ~rem(timeElapsed, T) * timeElapsed / T;
    nSampleSkipped = round(timeElapsed / T) - 1;
    if mod == 0
        % do another cross check and see if within tolerance
        if ~allclose(((timeElapsed / T) - 1) , nSampleSkipped, ...
                'atol', TIME_COMPARISON_TOL)
            warning('Inexact number of missing samples detected in NaN filling')
            boolFsInconsistent = true;
        end
    end
    assert(nSampleSkipped > 0, 'NaN filling pipeline in preprocessing failed')
    
    % append outer structure with data before filling and the necessary
    % NaN paddings
    timewNaN = [timewNaN; time(idxStart:idxEnd)];
    timeRelwNaN = [timeRelwNaN; timeRel(idxStart:idxEnd)];
    datawNaN = [datawNaN; data(idxStart:idxEnd, :)];

    timeNaNPadCurr = seconds((1:nSampleSkipped) * T)' + time(idxEnd);
    timeRelNaNPadCurr = ((1:nSampleSkipped) * T)' + timeRel(idxEnd);
    dataNaNPadCurr = ones(nSampleSkipped, size(data, 2)) * NaN;

    timewNaN = [timewNaN; timeNaNPadCurr];
    timeRelwNaN = [timeRelwNaN; timeRelNaNPadCurr];
    datawNaN = [datawNaN; dataNaNPadCurr];
end

% have to pad last discontinuity to end
timewNaN = [timewNaN; time((idxSkip(end) + 1):end)];
timeRelwNaN = [timeRelwNaN; timeRel((idxSkip(end) + 1):end)];
datawNaN = [datawNaN; data((idxSkip(end) + 1):end, :)];


%% Sanity check for output

% assert that start and end times are not changed
assert(timewNaN(1) == time(1) && timewNaN(end) == time(end), 'Inconsistent start and end times after NaN filling')

% assert that only one sample period exists
if ~boolFsInconsistent
    uniqueT = unique(seconds(diff(timewNaN)));
    assert(all(abs(uniqueT - T) < TIME_COMPARISON_TOL), 'Time comparison fails')
end

% assert that same amount of actual data points are in the data
idxValid = ~any(isnan(datawNaN), 2);
assert(sum(idxValid) == size(data, 1), 'Incorrect number of valid data points after padding');
assert(size(datawNaN, 2) == size(data, 2), 'Incorrect number of channels after padding');
assert(allclose(datawNaN(idxValid, :), data), 'These should be the same');


end