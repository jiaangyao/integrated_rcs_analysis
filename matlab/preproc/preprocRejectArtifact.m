function rawDataStructOut = preprocRejectArtifact(...
    rawDataStructIn, metaDataIn, preprocCfg)
%% Preprocessing wrapper script - perform artifact rejection
% Perform artifact rejection using amplitude thresholding right now
% 
% INPUT:
% rawDataStructIn             - struct: struct holding all time domain data
% metaDataIn                  - struct: most general metadata struct
% preprocCfg                  - struct: preprocessing related configuration


%% Perform amplitude thresholding-based artifact rejection

% unpack the data struct


% create the empty variables holding all values

% note that data right now is in (time, channel)
relTimewNAN = NaN(size(relTime));
absTimewNAN = NaT(size(absTime), 'TimeZone', absTime.TimeZone);
rawDatawNAN = NaN(size(rawData));
rawDataFiltwNAN = NaN(size(rawDataFilt));
nSample = size(rawData, 1);

% compute the start of each epoch
idxStart = 1:round(cfg.len_epoch_prune * fs):(nSample - 1);
idxStart = [idxStart, nSample];

idxValidEpoch = [];
timeStart = [];

% subsequently remove data chunks with stimulation artifact
for i = 1:(numel(idxStart)-1)
    % current start and end
    currIdxStart = idxStart(i);
    currIdxEnd = idxStart(i + 1) - 1;

    % obtain the data corresponding to current consecutive time
    currRelTime = relTime(currIdxStart: currIdxEnd);
    currAbsTime = absTime(currIdxStart: currIdxEnd);
    timeStart = [timeStart, currAbsTime(1)];
    
    currEpoch = rawData(currIdxStart: currIdxEnd, :);
    currEpochFilt = rawDataFilt(currIdxStart: currIdxEnd, :);

    % check for amplitude for artifacts
    if any(max(abs(currEpochFilt), [], 1) > cfg.amp_thres)
        idxValidEpoch = [idxValidEpoch, 0];
        continue;
    end
    
    % sanity check
    if any(isnan(currEpoch))
        idxValidEpoch = [idxValidEpoch, 0];
        continue;
    end

%     % TODO: check for missing chunks of data and fill with nan if necessary
%     if any(diff(curr_t) > 2)
%         error("NotImplementedError")
%     end
    
    relTimewNAN(currIdxStart: currIdxEnd) = currRelTime;
    absTimewNAN(currIdxStart: currIdxEnd) = currAbsTime;
    rawDatawNAN(currIdxStart: currIdxEnd, :) = currEpoch;
    rawDataFiltwNAN(currIdxStart: currIdxEnd, :) = currEpochFilt;
    idxValidEpoch = [idxValidEpoch, 1];

end

% add final time stamp
timeStart = [timeStart, absTime(end)];

end