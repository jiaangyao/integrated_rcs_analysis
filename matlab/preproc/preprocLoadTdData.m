function rawDataStructOut = preprocLoadTdData(timeDomainData, ...
    metaDataIn, preprocCfg)
%% Preprocessing wrapper script - loading of stim meta data
% Parses timeDomainData and loads in time and raw data information
% all times in seconds and neural data in uV
% 
% INPUT:
% timeDomainData              - table: time domain data table from processRCS
% metaDataIn                  - struct: most general metadata struct
% preprocCfg                  - struct: preprocessing related configuration


%% Magic number definition

TIME_CONVERSION_FACTOR = 1e-3;
VOLTAGE_CONVERSION_FACTOR = 1e3;
TIME_COMPARISON_TOL = 1e-3;


%% Loads all time domain data

% obtain the sampling rate and the time variables
fs = metaDataIn.timeDomainMetaData.fs;
time = timeDomainData.localTime;
timeRel = TIME_CONVERSION_FACTOR * (timeDomainData.DerivedTime - ...                             % normalize to s
    timeDomainData.DerivedTime(1));

% next extract all the data in format time*channel
rawData = timeDomainData{:, {'key0', 'key1', 'key2', 'key3'}};
rawData = rawData * VOLTAGE_CONVERSION_FACTOR;                                                   % normalize to uV

% then filter the signal for subcortical channels
vecFiltSubCortical = signalGetSubCorticalFilt(metaDataIn, preprocCfg);
boolSubCortical = metaDataIn.timeDomainMetaData.subCortical.boolLocChan;
rawDataSubCortical = rawData(:, boolSubCortical);
rawDataSubCorticalFilt = signalFiltRawData(rawDataSubCortical, fs, ...
    vecFiltSubCortical);

% next filter the signal for cortical channels
vecFiltCortical = signalGetCorticalFilt(metaDataIn, preprocCfg);
boolCortical = metaDataIn.timeDomainMetaData.cortical.boolLocChan;
rawDataCortical = rawData(:, boolCortical);
rawDataCorticalFilt = signalFiltRawData(rawDataCortical, fs, ...
    vecFiltCortical);

% concatenate to form the filtered data
rawDataFilt = [rawDataSubCorticalFilt, rawDataCorticalFilt];


%% also generate arrays that are nan-filled

% process array and fill missing data with NaNs
[timewNaN, timeRelwNaN, rawDatawNaN] = varDimFillwithNaN(time, ...
    timeRel, rawData, fs, TIME_COMPARISON_TOL);

[~, ~, rawDataFiltwNaN] = varDimFillwithNaN(time, ...
    timeRel, rawDataFilt, fs, TIME_COMPARISON_TOL);


%% append everything to output struct

% append all relevant information
rawDataStructOut.time = time;
rawDataStructOut.timeRel = timeRel;
rawDataStructOut.timewNaN = timewNaN;
rawDataStructOut.timeRelwNaN = timeRelwNaN;

rawDataStructOut.data = rawData;
rawDataStructOut.dataFilt = rawDataFilt;
rawDataStructOut.datawNaN = rawDatawNaN;
rawDataStructOut.dataFiltwNaN= rawDataFiltwNaN;


end