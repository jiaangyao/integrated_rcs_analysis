function metaDataStructOut = preprocLoadMetaDataPower(powerData, ...
    powerSettings, metaDataIn)
%% Preprocessing wrapper script - loading of relevant power metadata
% Loads metadata pertaining to the power domain data such as frequency range
% of power bands from powerSettings
% 
% INPUT:
% powerData             - table: power domain data table from processRCS
% powerSettings         - table: power domain setting table from processRCS
% metaDataIn            - struct: most general metadata struct


%% Check for whether power setting is enabled

% check for whether or not these settings are enabled
if size(powerSettings, 1) > 1
    metaDataStructOut.isEnabled = true;
else
    metaDataStructOut.isEnabled = false;
    return
end


%% Obtain the sampling rate for the power data

% estimate the sampling rate from streamed data
metaDataStructOut.fs = getSamplingRatefromData(powerData);

% check if FFT overlap sample is integer
if ~metaDataIn.fftMetaData.boolInExactFs
    % sampling rate is inexact then carry parameters over from FFT
    metaDataStructOut.boolInExactFs = true;
    metaDataStructOut.vecPowerWin = metaDataIn.fftMetaData.vecFFTWin;
    metaDataStructOut.vecPowerWinSample = metaDataIn.fftMetaData.vecFFTWinSample;
else
    metaDataStructOut.boolInExactFs = false;
    metaDataStructOut.vecPowerWin = [];
    metaDataStructOut.vecPowerWinSample= [];
end


%% Parse which power bands are beta and gamma respectively

% loop through all power bands
currPowerBandsSetting = powerSettings(end, :).powerBands;
boolVecBetaRangePB = [];
boolVecGammaRangePB = [];
strVecChanwPB = {};
for i = 1:8     % know at most can stream 8 powerbands at once
    % obtain right channel name
    idxChan = floor((i + 1) / 2);
    strVecChanwPB{i} = metaDataIn.timeDomainMetaData.vecStrChan{idxChan};

    % check if beta
    boolCurrPBBeta = currPowerBandsSetting.lowerBound(i) >= 13 && ...
        currPowerBandsSetting.upperBound(i) <= 30;
    boolVecBetaRangePB = [boolVecBetaRangePB; boolCurrPBBeta];

    % check if gamma
    boolCurrPBGamma = currPowerBandsSetting.lowerBound(i) >= 50 && ...
        currPowerBandsSetting.upperBound(i) <= 120;
    boolVecGammaRangePB = [boolVecGammaRangePB; boolCurrPBGamma];
end

% pend to output structure
metaDataStructOut.boolVecBetaRangePB = boolVecBetaRangePB;
metaDataStructOut.boolVecGammaRangePB = boolVecGammaRangePB;
metaDataStructOut.strVecChanwPB = varDimConvertWideArraytoTall(strVecChanwPB);
metaDataStructOut.powerBandsInHz = currPowerBandsSetting.powerBandsInHz;
metaDataStructOut.powerBandSetting = currPowerBandsSetting;


%% Parsing of shift information

% also parse shift of powerbands
currFFTConfig = powerSettings(end, :).fftConfig;
strShift = regexp(currFFTConfig.bandFormationConfig, '[Ss]hift[0-8]', ...
    'match');
if numel(strShift) == 1
    if strcmpi(strShift{1}, 'Shift0')
        idxShift = 0;
    elseif strcmpi(strShift{1}, 'Shift1')
        idxShift = 1;
    elseif strcmpi(strShift{1}, 'Shift2')
        idxShift = 2;
    elseif strcmpi(strShift{1}, 'Shift3')
        idxShift = 3;
    elseif strcmpi(strShift{1}, 'Shift4')
        idxShift = 4;
    elseif strcmpi(strShift{1}, 'Shift5')
        idxShift = 5;
    elseif strcmpi(strShift{1}, 'Shift6')
        idxShift = 6;
    elseif strcmpi(strShift{1}, 'Shift7')
        idxShift = 7;      
    elseif strcmpi(strShift{1}, 'Shift8')
        idxShift = 8;
    else
        warning("Unknown shift extracted")
        idxShift = NaN;
    end
elseif numel(strShift) > 1
    warning("More than one shift extracted");
    idxShift = NaN;
else
    warning('Extraction of shift information failed');
    idxShift = NaN;
end

metaDataStructOut.fftConfig = currFFTConfig;
metaDataStructOut.idxShift = idxShift;


%% Also backup the original metadata

% add local time stamps to the table
powerSettingswTime = dtAppendDateTime2Table(powerSettings, ...
    {'timeStart', 'timeStop'}, metaDataIn.genMetaData.timeFormat, ...
    metaDataIn.genMetaData.timeZone);
metaDataStructOut.origMetaData = powerSettingswTime;


end