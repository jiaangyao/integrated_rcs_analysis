function fs = getSamplingRatefromData(data)
%% getDataUtils script - estimate sampling rate information from data
% Loads sampling rate information and optionally compare with time domain
% data to perform sanity checks regarding the validity of the sampling rate
% 
% INPUT:
% data                  - table: power/adaptive domain data table from processRCS


%% Estimate sampling rate based on streamed data

% obtain the valid entries from the data table
idxValidSamplingRate = ~isnan(data.samplerate);

% next find out the number of unique elements in the streamed data
vecUniqueSamplingRate = ...
    unique(data.samplerate(idxValidSamplingRate));
if numel(vecUniqueSamplingRate) > 1
    error('Multiple sampling rates exist in the current session based on timeDomainData')
end

% obtain the actual sampling rate
fs = vecUniqueSamplingRate(1);


end