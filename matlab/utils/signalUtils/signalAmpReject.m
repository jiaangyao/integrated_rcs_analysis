function [datawAmpRej, timeStart, idxValidEpoch] = signalAmpReject(time, ...
    data, timeRes, fs, thresh)
%% signalUtils script - perform amplitude-based thresholding for rejection
% Perform amplitude based thresholding (after DC component removal) for
% data and outputs both the cleaned data and the boolean mask for cleaned
% data
%
% INPUT:
% rawData               - double: raw data in format (time*channel)
% fs                    - double: sampling rate
% filtOrder             - double: order for filter
% cutoffFreqLow         - double: lower 6-dB frequency
% cutoffFreqHigh        - double: higher 6-dB frequency


%% perform amplitude thresholding

% create empty data structure for holding output data
datawAmpRej = NaN(size(data));
idxValidEpoch = [];

% compute the start index of each epoch
nSample = size(data, 1);
idxStart = 1:round(timeRes * fs):(nSample - 1);
idxStart = [idxStart, nSample];

idxValidEpoch = [];
timeStart = [];

% subsequently remove data chunks with stimulation artifact
for i = 1:(numel(idxStart)-1)
    % current start and end
    currIdxStart = idxStart(i);
    currIdxEnd = idxStart(i + 1) - 1;

    % obtain the data corresponding to current consecutive time
    currTime = time(currIdxStart: currIdxEnd);
    timeStart = [timeStart, currTime(1)];
    currEpoch = data(currIdxStart: currIdxEnd, :);

    % check for amplitude for artifacts
    if any(max(abs(currEpoch), [], 1) > thresh)
        idxValidEpoch = [idxValidEpoch, 0];
        continue;
    end

    % sanity check
    if any(isnan(currEpoch))
        idxValidEpoch = [idxValidEpoch, 0];
        continue;
    end
end
end