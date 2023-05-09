function [relTimewNAN, absTimewNAN, rawDatawNAN] = ...
    preprocArtifactRejectwIdx(relTime, absTime, rawData, ...
    timeStart, idxValidEpoch_CORR)

% create the empty variables for holding data
relTimewNAN = NaN(size(relTime));
absTimewNAN = NaT(size(absTime), 'TimeZone', absTime.TimeZone);
rawDatawNAN = NaN(size(rawData));

% now also perform artifact rejection based previously calculated idx
for i = 1:(numel(timeStart) - 1)
    % obtain matching indices
    idxDataCurr = absTime >= timeStart(i) & absTime < timeStart(i + 1);
    
    % obtain data from current epoch
    currRelTime = relTime(idxDataCurr);
    currAbsTime = absTime(idxDataCurr);
    currEpoch = rawData(idxDataCurr, :);

    % check for amplitude for artifacts
    if ~idxValidEpoch_CORR(i)
        continue;
    end

%     % TODO: check for missing chunks of data and fill with nan if necessary
%     if any(diff(curr_t) > 2)
%         error("NotImplementedError");
%     end
    
    relTimewNAN(idxDataCurr) = currRelTime;
    absTimewNAN(idxDataCurr) = currAbsTime;
    rawDatawNAN(idxDataCurr, :) = currEpoch;
end

end