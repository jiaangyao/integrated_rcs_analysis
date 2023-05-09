function rawDataFilt = signalFiltRawData(rawData, fs, vecFilt)
%% signalUtils script - filter raw time domain data from subcortical/cortical region
% Processes the sub-cortical/cortical data based on user-provided filters
%
% INPUT:
% rawData               - double: raw data from STN/GPi or cortex in format (time*channel)
% fs                    - double: sampling rate
% vecFilt               - cell: list of filter objects


%% Now process by filtering the various data

% loop through the filter objects and apply the filter recursively
rawDataFilt = rawData(:, :);
for i = 1:numel(vecFilt)
    % obtain current filter
    filtCurr = vecFilt{i};

    % get name of filter and apply it based on type
    if strcmpi(filtCurr.strFiltName, 'lowpass')
        % lowpass filter the data
        assert(numel(filtCurr.cutoffFreq) == 1, 'Length of filter edge should be 1')
        rawDataFilt = signalFiltLowpass(rawDataFilt, ...
            filtCurr.filtOrder, filtCurr.cutoffFreq, fs);

    elseif strcmpi(filtCurr.strFiltName, 'highpass')
        % highpass filter the data
        assert(numel(filtCurr.cutoffFreq) == 1, 'Length of filter edge should be 1')
        rawDataFilt = signalFiltHighpass(rawDataFilt, ...
            filtCurr.filtOrder, filtCurr.cutoffFreq, fs);

    elseif strcmpi(filtCurr.strFiltName, 'bandpass')
        % bandpass filter the data
        assert(numel(filtCurr.cutoffFreq) == 2, 'Length of filter edge should be 2')
        rawDataFilt = signalFiltBandpass(rawDataFilt, ...
            filtCurr.filtOrder, filtCurr.cutoffFreq(1), ...
            filtCurr.cutoffFreq(2), fs);

    elseif strcmpi(filtCurr.strFiltName, 'bandstop')
        % bandpass filter the data
        assert(numel(filtCurr.cutoffFreq) == 2, 'Length of filter edge should be 2')
        rawDataFilt = signalFiltBandstop(rawDataFilt, ...
            filtCurr.filtOrder, filtCurr.cutoffFreq(1), ...
            filtCurr.cutoffFreq(2), fs);

    else
        error('Unknown filter type requested')
    end
end


end