function metaDataStructOut = SP_preprocAllMetaData(metaData, timeDomainData, ...
    timeDomainSettings, fftData, fftSettings, powerData, powerSettings, ...
    adaptiveData, adaptiveEmbeddedRuns_StimSettings, detectorSettings, ...
    stimLogSettings, eventLogTable, pathIn, bool_aDBS, varargin)
%% Preprocessing wrapper script - loading of all metadata
% General wrapper for various scripts that loads all relevant metadata
% pertaining to data analysis and processing
% 
% INPUT:
% metaData                           - struct: metadata structure from 
%                                           processRCS
% timeDomainData                     - table: time domain data table from 
%                                           processRCS
% timeDomainSettings                 - table: time domain setting table 
%                                           from processRCS
% fftData                            - table: FFT data table from 
%                                           processRCS (currently not used)
% fftSettings                        - table: FFT setting table from
%                                           processRCS (currently not used)
% powerData                          - table: power data table from 
%                                           processRCS
% powerSettings                      - table: power domain setting table 
%                                           from processRCS
% adaptiveData                       - table: adaptive data table from 
%                                           processRCS
% adaptiveEmbeddedRuns_StimSettings  - table: adaptive setting table from 
%                                           processRCS
% detectorSettings                   - table: LD setting table from
%                                           processRCS
% stimLogSettings                    - table: stim setting table from
%                                           processRCS
% eventLogTable                      - table: event table from processRCS
% pathIn                             - string: input full absolute path
% bool_aDBS                          - boolean: whether or not current 
%                                           recording is from aDBS
% 
% OPTIONAL INPUT:
% strStep                            - string: current step in aDBS pipeline
%                                           default: ''
% strRound                           - string: current round in aDBS pipeline
%                                           default: ''
% boolCheckTDData                    - boolean: whether to use time domain
%                                           data to verify sample rate
%                                           default: false
% boolReverseNonActiveLD             - boolean: whether to reverse rule for
%                                           LD -> high LD == low stim
%                                           default: false
% boolGroupDOnly                     - boolean: whether to only use group D
%                                           default: false
% boolCheckRampTime                  - boolean: whether to check for ramping
%                                           default: false
% stimAmp                            - double: if check ramping desired stim
%                                           default: 0


%% Input parsing
% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

% optional commands for handling the sampling rate parsing
addParameter(p, 'strStep', '', ...
    @(x) validateattributes(x, {'char'}, {'nonempty'}));
addParameter(p, 'strRound', '', ...
    @(x) validateattributes(x, {'char'}, {'nonempty'}));
addParameter(p, 'boolCheckTDData', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'boolReverseNonActiveLD', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'boolGroupDOnly', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'boolCheckRampTime', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'stimAmp', 0, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
strStep = p.Results.strStep;
strRound = p.Results.strRound;
boolCheckTDData = p.Results.boolCheckTDData;
boolReverseNonActiveLD = p.Results.boolReverseNonActiveLD;
boolGroupDOnly = p.Results.boolGroupDOnly;
boolCheckRampTime = p.Results.boolCheckRampTime;
stimAmp = p.Results.stimAmp;


%% Parsing the most general metadata first

% creates the metadata and populate with demographic and timing related 
% information
metaDataStructOut.genMetaData = preprocLoadMetaDataGen(metaData, ...
    timeDomainData, pathIn, bool_aDBS, 'strStep', strStep, ...
    'strRound', strRound);


%% Parsing the time domain metadata

% parse timeDomainSettings and populate metadata struct with relevant
% information
metaDataStructOut.timeDomainMetaData = preprocLoadMetaDataTime(...
    boolCheckTDData, timeDomainData, timeDomainSettings, ...
    metaDataStructOut);


%% Parsing the FFT domain metadata

% parse fftSettings and populate metadata struct with relevant
% information
metaDataStructOut.fftMetaData = preprocLoadMetaDataFFT(fftData, fftSettings, ...
    metaDataStructOut);


%% Parsing the power domain related metadata

% parse powerSettings and populate metadata struct with relevant
% information
metaDataStructOut.powerMetaData = preprocLoadMetaDataPower(powerData, ...
    powerSettings, metaDataStructOut);


%% Parsing the adaptive related metadata

% parse AdaptiveEmbeddedRuns_StimSettings and DetectorSettings and
% populate metadata struct with relevant information
metaDataStructOut.adaptiveMetaData = preprocLoadMetaDataAdaptive(...
    adaptiveData, adaptiveEmbeddedRuns_StimSettings, detectorSettings, ...
    metaDataStructOut, boolReverseNonActiveLD);


%% Parsing the stim related information

% parse the stim related metadata
metaDataStructOut.stimMetaData = preprocLoadMetaDataStim(stimLogSettings, ...
    metaDataStructOut, 'boolGroupDOnly', boolGroupDOnly, ...
    'boolCheckRampTime', boolCheckRampTime, 'stimAmp', stimAmp);


%% Parsing the patient self reported disease markers

% load patient self-reported markers if adaptive is on
metaDataStructOut.eventMarker = preprocLoadMetaDataEventMarker(...
    eventLogTable, metaDataStructOut);


end