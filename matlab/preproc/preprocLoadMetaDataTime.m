function metaDataStructOut = preprocLoadMetaDataTime(boolCheckTDData, ...
    timeDomainData, timeDomainSettings, metaDataIn)
%% Preprocessing wrapper script - loading of relevant time domain metadata
% Loads metadata pertaining to the time domain data such as sampling rate
% channel names from the RCS timeDomainSettings struct
% 
% INPUT:
% boolCheckTDData       - bool: bool for whether to check with TD data
% timeDomainData        - table: time domain data table from processRCS
% timeDomainSettings    - table: time domain setting table from processRCS
% metaDataIn            - struct: most general metadata struct


%% Initial sanity checks

% first confirm that timeDomainData is tall array
assert(varDimCheckValidEntryExist(timeDomainData))
assert(varDimCheckTallArray2D(timeDomainData));

%% Parsing the sampling rate in the original data

% parse the sampling rate based on timeDomainSettings and user provided
% input
metaDataStructOut.fs = getSamplingRate(boolCheckTDData, ...
    timeDomainData, timeDomainSettings);


%% Parsing the channel name in the original data

% get the channel names
metaDataStructOut.vecStrChan = getChannelNames(timeDomainSettings);

% figure out cortical and subcortical boolean masks
[metaDataStructOut.subCortical, metaDataStructOut.cortical] ...
    = strParExtractChanInfo(metaDataStructOut.vecStrChan);


%% Parsing of additional information

% append the minus and the plus electrodes for referencing
% here can access the last setting directly since consistency of channel
% names has been checked in getChannelNames
metaDataStructOut.minusInput = varDimConvertWideArraytoTall(...
    {timeDomainSettings(end, :).TDsettings{1}.minusInput});
metaDataStructOut.plusInput = varDimConvertWideArraytoTall(...
    {timeDomainSettings(end, :).TDsettings{1}.plusInput});


%% Also backup the original metadata

% add local time stamps to the table
timeDomainSettingswTime = dtAppendDateTime2Table(timeDomainSettings, ...
    {'timeStart', 'timeStop'}, metaDataIn.genMetaData.timeFormat, ...
    metaDataIn.genMetaData.timeZone);
metaDataStructOut.origMetaData = timeDomainSettingswTime;


end