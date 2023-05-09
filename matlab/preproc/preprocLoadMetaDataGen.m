function metaDataStructOut = preprocLoadMetaDataGen(metaData, ...
    timeDomainData, pathIn, bool_aDBS, varargin)
%% Preprocessing wrapper script - loading of relevant meta metadata
% Loads relevant subject information and time information such as the 
% start and end of current session and obtains the time format and time 
% zone of the localTime information stored in RCS
% Note: creates the return metastructure
% 
% INPUT:
% metaData              - struct: metadata structure from processRCS
% timeDomainData        - table: time domain data table from processRCS
% pathIn                - string: absolute path to the current session file
% bool_aDBS             - boolean: whether or not current recording is from aDBS
% 
% 
% OPTIONAL INPUT:
% strStep               - string: current step in aDBS pipeline
%                           default: ''
% strRound              - string: current round in aDBS pipeline
%                           default: ''


%% Input parsing
% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'strStep', '', ...
    @(x) validateattributes(x, {'char'}, {'nonempty'}));
addParameter(p, 'strRound', '', ...
    @(x) validateattributes(x, {'char'}, {'nonempty'}));

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
strStep = p.Results.strStep;
strRound = p.Results.strRound;


%% Parsing of demographic information

% parse the subject number and side of the electrode
metaDataStructOut.subjectIDFull = metaData.subjectID;
metaDataStructOut.strSubject = metaData.subjectID(1:end - 1);
metaDataStructOut.strSide = metaData.subjectID(end);
if strcmpi(metaDataStructOut.strSide, 'L')
    metaDataStructOut.strSideFull = 'Left';
elseif strcmpi(metaDataStructOut.strSide, 'R')
    metaDataStructOut.strSideFull = 'Right';
else
    error('Incorret parsing of electrode implant side')
end
metaDataStructOut.strHandedness = metaData.handedness;

% next try to obtain the device ID based on the subject
metaDataStructOut.deviceID = getDeviceName(metaDataStructOut.strSubject, ...
    metaDataStructOut.strSide);
metaDataStructOut.pathIn = pathIn;

% get the sub-cortical lead target
metaDataStructOut.leadTarget = metaData.leadTargets{1};


%% Parsing of device specific information

% unpack the ampGains struct
vecAmpGains = cellfun(@(x)(metaData.ampGains.(x)), ...
    fieldnames(metaData.ampGains));
metaDataStructOut.ampGains = vecAmpGains;


%% Parsing of all time related information

% obtain absolute time of recording start and end
metaDataStructOut.sessionStartDT = timeDomainData.localTime(1);
metaDataStructOut.sessionEndDT = timeDomainData.localTime(end);

metaDataStructOut.sessionStartUnix = timeDomainData.DerivedTime(1);
metaDataStructOut.sessionEndUnix = timeDomainData.DerivedTime(end);

% obtain the year, month and day of session
metaDataStructOut.sessionYear = metaDataStructOut.sessionStartDT.Year;
metaDataStructOut.sessionMonth = metaDataStructOut.sessionStartDT.Month;
metaDataStructOut.sessionDay = metaDataStructOut.sessionStartDT.Day;

% other metadata
metaDataStructOut.timeFormat = metaDataStructOut.sessionStartDT.Format;
metaDataStructOut.timeZone = metaDataStructOut.sessionStartDT.TimeZone;

% check if session is AM or PM based on start time
metaDataStructOut.sessionAMorPM = dtCheckAMorPM(metaDataStructOut.sessionStartDT, ...
    metaDataStructOut);

% log the time of this processing for versioning purposes
metaDataStructOut.timeNow = datetime;
metaDataStructOut.timeNow.Format = metaDataStructOut.timeFormat;
metaDataStructOut.timeNow.TimeZone = metaDataStructOut.timeZone;


%% check if aDBS and also add in some meta data info

% log the aDBS related meta info
metaDataStructOut.bool_aDBS = bool_aDBS;
metaDataStructOut.strStep = strStep;
metaDataStructOut.strRound = strRound;


%% Also backup the original metadata

metaDataStructOut.origMetaData = metaData;


end