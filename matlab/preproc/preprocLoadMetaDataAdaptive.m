function metaDataStructOut = preprocLoadMetaDataAdaptive(adaptiveData, ...
    adaptiveEmbeddedRuns_StimSettings, detectorSettings, metaDataIn, ...
    boolReverseNonActiveLD)
%% Preprocessing wrapper script - loading of relevant adaptive metadata
% Loads metadata pertaining to the adaptive data separately for LD0 and 
% LD1 based off adaptiveEmbeddedRuns_StimSettings and DdetectorSettings 
% 
% INPUT:
% adaptiveData                       - table: adaptive domain data table 
%                                           from processRCS
% adaptiveEmbeddedRuns_StimSettings  - table: adaptive setting table from 
%                                           processRCS
% detectorSettings                   - table: LD setting table from
%                                           processRCS
% metaDataIn                         - struct: most general metadata struct
% boolReverseNonActiveLD             - boolean: whether to reverse rule for
%                                           LD -> high LD == low stim

%% Obtain the sampling rate for the adaptive data

% estimate the sampling rate from streamed data
metaDataStructOut.fs = getSamplingRatefromData(adaptiveData);

% check if FFT overlap sample is integer
if ~metaDataIn.powerMetaData.boolInExactFs
    % sampling rate is inexact then carry parameters over from FFT
    metaDataStructOut.boolInExactFs = true;
    metaDataStructOut.vecAdaptiveWin = ...
        metaDataIn.powerMetaData.vecPowerWin;
    metaDataStructOut.vecAdaptiveWinSample = ...
        metaDataIn.powerMetaData.vecPowerWinSample;
else
    metaDataStructOut.boolInExactFs = false;
    metaDataStructOut.vecAdaptiveWin = [];
    metaDataStructOut.vecAdaptiveWinSample = [];
end


%% Check for whether adaptive and detector setting is enabled

% first check if adaptive stim setting is enabled
if size(adaptiveEmbeddedRuns_StimSettings, 1) > 0
    metaDataStructOut.isEnabledAdaptive = true;
else
    metaDataStructOut.isEnabledAdaptive = false;
end

% next check for whether detector setting is enabled
if size(detectorSettings, 1) > 0
    Ld0Enabled = contains(detectorSettings(end, :).Ld0.detectionInputs_BinaryCode, '1');
    Ld1Enabled = contains(detectorSettings(end, :).Ld1.detectionInputs_BinaryCode, '1');
    if Ld0Enabled || Ld1Enabled
        metaDataStructOut.isEnabledDetector = true;
    else
        metaDataStructOut.isEnabledDetector = false;
    end
else
    metaDataStructOut.isEnabledDetector = false;
end

% if either is not enabled then adaptive stim is not enabled
if ~(metaDataStructOut.isEnabledAdaptive && metaDataStructOut.isEnabledDetector)
    metaDataStructOut.isEnabled = false;
    return
else
    metaDataStructOut.isEnabled = true;
end


%% Parsing stim and state information

% obtain the LD information based on adaptive information from latest
% adaptive setting
currAdaptiveSetting = adaptiveEmbeddedRuns_StimSettings(end, :);
currStateTable = currAdaptiveSetting.states;

% obtain the current detector setting
currDetectorSetting = detectorSettings(end, :);

% parse stim amplitude
fullStimAmp = [];
for i = 0:8         % know there are 9 states in total
    currStrStimAmp = sprintf('state%d_AmpInMilliamps', i);
    currStimAmp = currStateTable.(currStrStimAmp);
    fullStimAmp = [fullStimAmp; currStimAmp];
end
fullStimAmp = reshape(fullStimAmp(:, 1), [3, 3])';
idxValidStim = fullStimAmp ~= -1;
fullStimValid = fullStimAmp;
fullStimValid(~idxValidStim(:, 1), :) = [];
fullStimValid(:, ~idxValidStim(1, :)) = [];

% also get the state table
fullStateName = reshape(0:8, [3, 3])';
fullStateNameValid = fullStateName;
fullStateNameValid(~idxValidStim(:, 1), :) = [];
fullStateNameValid(:, ~idxValidStim(1, :)) = [];

% append to output variable now
metaDataStructOut.fullStimAmp = fullStimAmp;
metaDataStructOut.idxValidStim = idxValidStim;
metaDataStructOut.fullStimValid = fullStimValid;

metaDataStructOut.fullStateName = fullStateName;
metaDataStructOut.fullStateNameValid = fullStateNameValid;


%% parse various LD0 parameters

% obtain the information struct for LD0
metaDataStructOut.LD0 = cfgGetLDInfo('LD0', currAdaptiveSetting, ...
    currDetectorSetting, metaDataStructOut, metaDataIn, boolReverseNonActiveLD);


%% parse various LD1 parameters

% obtain the information struct for LD1
metaDataStructOut.LD1 = cfgGetLDInfo('LD1', currAdaptiveSetting, ...
    currDetectorSetting, metaDataStructOut, metaDataIn, boolReverseNonActiveLD);


%% if fake adaptive days then change the stim level

% if only one stim amplitude for both LDs then fake adaptive protocol


if (numel(unique(metaDataStructOut.LD0.stimLevel)) == ...
        numel(unique(metaDataStructOut.LD1.stimLevel))) && ...
        (numel(unique(metaDataStructOut.LD0.stimLevel)) == 1)
    if strcmp(metaDataIn.genMetaData.strSubject, 'RCS02')
        error("NotImplementedError");

    elseif strcmp(metaDataIn.genMetaData.strSubject, 'RCS14')
        warning('Hardcoded stim level for fake adaptive data')
        LD0.stimLevel = [3.1, 3.7];
        LD1.stimLevel = [3.1, 3.7];
    end
else

end


%% add params common to both sides

% append to output structure
metaDataStructOut.LD0 = LD0;
metaDataStructOut.LD1 = LD1;


%% Also backup the original metadata

% add local time stamps to the adaptive setting table
adaptiveSettingswTime = dtAppendDateTime2Table(adaptiveEmbeddedRuns_StimSettings, ...
    {'HostUnixTime'}, metaDataIn.genMetaData.timeFormat, ...
    metaDataIn.genMetaData.timeZone, 'boolAddStartorStop', true, ...
    'strAddStartorStop', 'stop', 'metaDataIn', metaDataIn);
metaDataStructOut.origMetaData.adaptiveSettings = adaptiveSettingswTime;

% add local time stamps to the detector setting table
detectorSettingswTime = dtAppendDateTime2Table(detectorSettings, ...
    {'HostUnixTime'}, metaDataIn.genMetaData.timeFormat, ...
    metaDataIn.genMetaData.timeZone, 'boolAddStartorStop', true, ...
    'strAddStartorStop', 'stop', 'metaDataIn', metaDataIn);
metaDataStructOut.origMetaData.detectorSettings = detectorSettingswTime;


end