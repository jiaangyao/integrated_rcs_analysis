function metaDataStructOut = preprocLoadMetaDataStim(stimLogSettings, ...
    metaDataIn, varargin)
%% Preprocessing wrapper script - loading of stim meta data
% Parses stimLogSettings and look for relevant information such as times in
% desired stimulation
% 
% INPUT:
% stimLogSettings             - table: event log table from processRCS
% metaDataIn                  - struct: most general metadata struct
% 
% 
% OPTIONAL INPUT:
% boolGroupDOnly              - boolean: whether to only use group D
%                                   default: false
% boolCheckRampTime           - boolean: whether check for ramp time
%                                   default: false
% stimAmp                     - double: desired amplitude for stimulation
%                                   default: 0


%% Input parsing
% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

% optional commands for handling the sampling rate parsing
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
boolGroupDOnly = p.Results.boolGroupDOnly;
boolCheckRampTime = p.Results.boolCheckRampTime;
stimAmp = p.Results.stimAmp;


%% Parse the stimLogSettings and parse the current stim amplitude

% first add the time information to the table
stimLogSettingswTime = dtAppendDateTime2Table(stimLogSettings, ...
    {'HostUnixTime'}, metaDataIn.genMetaData.timeFormat, ...
    metaDataIn.genMetaData.timeZone, 'boolAddStartorStop', true, ...
    'strAddStartorStop', 'stop', 'metaDataIn', metaDataIn);

% next redact entries where therapy status is off
idxValidTherapy = stimLogSettingswTime.therapyStatus == 1;
stimLogSettingswTimeRedact = stimLogSettingswTime(idxValidTherapy, :);

% loop through all entries and parse directly for stim information
stimLogSettingswTimeRedactFull = table();
for i = 1:size(stimLogSettingswTimeRedact, 1)
    rowCurr = stimLogSettingswTimeRedact(i, :);
    appendRowCurr = rowCurr(:, :);

    % unpack and obtain the current stim amp, pulse width and stim rate
    appendRowCurr.('currGroupSetting') = rowCurr.(sprintf('Group%s', rowCurr.activeGroup{1}));
    progSplit = split(appendRowCurr.stimParams_prog1, ','); 
    appendRowCurr.('stimChan') = progSplit{1};
    appendRowCurr.('stimAmp') = appendRowCurr.currGroupSetting.ampInMilliamps(1);
    appendRowCurr.('pulseWidth') = appendRowCurr.currGroupSetting.pulseWidthInMicroseconds(1);
    appendRowCurr.('stimFreq') = appendRowCurr.currGroupSetting.RateInHz;
    
    % if need to check for ramp time
    if boolCheckRampTime
        appendRowCurr.('notRamping') = (appendRowCurr.('stimAmp') == stimAmp);
    end

    stimLogSettingswTimeRedactFull = ...
        [stimLogSettingswTimeRedactFull; appendRowCurr];
end

% if we only want Group D
if boolGroupDOnly
    idxValidGroupD = strcmpi(stimLogSettingswTimeRedactFull.activeGroup, 'D');
    stimLogSettingswTimeRedactFull = stimLogSettingswTimeRedactFull(idxValidGroupD, :);
end

% next also check if stim is enabled
if size(stimLogSettingswTimeRedactFull, 1) > 0
    metaDataStructOut.isEnabled = true;
end

% append to output structure
if metaDataStructOut.isEnabled
    metaDataStructOut.stimLogSettings = stimLogSettingswTimeRedactFull;
end

%% Also backup the original metadata

% add setting with datetime to output struct
metaDataStructOut.origMetaData = stimLogSettingswTime;


end