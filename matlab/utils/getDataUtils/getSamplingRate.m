function fs = getSamplingRate(boolCheckTDData, timeDomainData, ...
    timeDomainSettings)
%% getDataUtils script - loading of sampling rate information
% Loads sampling rate information and optionally compare with time domain
% data to perform sanity checks regarding the validity of the sampling rate
% 
% INPUT:
% boolCheckTDData       - bool: bool for whether to check with TD data
% timeDomainData        - table: time domain data table from processRCS
% timeDomainSettings    - table: time domain setting table from processRCS


%% Check in the time domain setting table

% first confirm uniform samping rate in the time domain setting table
if size(timeDomainSettings, 1) > 1
    if ~all(timeDomainSettings.samplingRate == ...
            timeDomainSettings.samplingRate(1))
        error('Multiple sampling rates exist in the current session based on timeDomainSettings')
    end
end

% load the sampling rate from the first row
fs = timeDomainSettings.samplingRate(1);


%% Perform check in data to confirm single sampling rate

% optionally go through actual data
if boolCheckTDData
    % obtain the valid entries from the data table
    idxValidSamplingRate = ~isnan(timeDomainData.samplerate);

    % next find out the number of unique elements in the streamed data
    vecUniqueSamplingRate = ...
        unique(timeDomainData.samplerate(idxValidSamplingRate));
    if numel(vecUniqueSamplingRate) > 1
        error('Multiple sampling rates exist in the current session based on timeDomainData')
    end

    % check to see if match sampling rate specified in setting
    if vecUniqueSamplingRate ~= fs
        error('Mismatch between sampling rate in timeDomainSettings and timeDomainData')
    end
end


end