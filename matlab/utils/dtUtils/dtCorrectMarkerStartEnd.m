function [vecMarkerStartCorr, vecMarkerEndCorr] = ...
    dtCorrectMarkerStartEnd(vecMarkerStart, vecMarkerEnd, varargin)
%% dtUtils script - correct for missing start and end markers
% Correct for loaded start and end markers for various events. Can either
% work by adding missing entries or deleting extra entries
%
% INPUT:
% vecMarkerStart        - datetime: array of start times
% vecMarkerEnd          - datetime: array of end times
% 
% 
% OPTIONAL INPUT:
% strAddorDrop          - string: whether the code add time or remove
%                           ['add'|'drop'] default: drop
% defaultDuration       - double: default duration of symptom in minute


%% Input parsing
% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

% optional commands for handling the sampling rate parsing
addParameter(p, 'strAddorDrop', 'drop', ...
    @(x) validateattributes(x, {'char'}, {'nonempty'}));
addParameter(p, 'defaultDuration', 0, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
strAddorDrop = p.Results.strAddorDrop;
defaultDuration = p.Results.defaultDuration;


%% Sanity check

% check operation mode is valid
assert(any(strcmpi(strAddorDrop, {'add', 'drop'})), 'Incorrect operation mode')
if strcmpi(strAddorDrop, 'add')
    assert(defaultDuration ~= 0, 'Need to provide a valid default duration')
end


%% Now Parse the two arrays

% create empty array for output
vecMarkerStartCorr = [];
vecMarkerEndCorr = [];

% start with the list with fewer entries
if numel(vecMarkerStart) < numel(vecMarkerEnd)
    for i = 1:numel(vecMarkerStart)
        % append to outer list first
        vecMarkerStartCorr = [vecMarkerStartCorr, vecMarkerStart(i)];

        % next try to find closet end time that's after start
        diffDysk = vecMarkerEnd - vecMarkerStart(i);
        minDiffDysk = min(diffDysk(diffDysk > 0));
        
        % append that to outer list
        vecMarkerEndCorr = [vecMarkerEndCorr, ...
            vecMarkerEnd(diffDysk == minDiffDysk)];
    end

    % if the operation mode is add missing entries
    if strcmpi(strAddorDrop, 'add')
        % first check which entries are dropped and obtain the
        % corresponding start times
        if numel(vecMarkerEndCorr) == 0
            vecMarkerEndSkipped = vecMarkerEnd;
        else
            vecMarkerEndSkipped = setdiff(vecMarkerEnd, vecMarkerEndCorr);
        end
        vecMarkerStartAdd = vecMarkerEndSkipped - minutes(defaultDuration);

        % next add these markers to the original list
        vecMarkerStartCorr = sort([vecMarkerStartCorr, vecMarkerStartAdd]);
        vecMarkerEndCorr = sort([vecMarkerEndCorr, vecMarkerEndSkipped]);
    end

elseif numel(vecMarkerStart) > numel(vecMarkerEnd)
    for i = 1:numel(vecMarkerEnd)
        % append to outer list first
        vecMarkerEndCorr = [vecMarkerEndCorr, vecMarkerEnd(i)];

        % next try to find closet start time that's before end
        diffDysk = vecMarkerStart - vecMarkerEnd(i);
        maxDiffDysk = max(diffDysk(diffDysk < 0));

        % append that to outer list
        vecMarkerStartCorr = [vecMarkerStartCorr, ...
            vecMarkerStart(diffDysk == maxDiffDysk)];
    end

    % if the operation mode is add missing entries
    if strcmpi(strAddorDrop, 'add')
        % first check which entries are dropped and obtain the
        % corresponding end times
        if numel(vecMarkerStartCorr) == 0
            vecMarkerStartSkipped = vecMarkerStart;
        else
            vecMarkerStartSkipped = setdiff(vecMarkerStart, vecMarkerStartCorr);
        end
        vecMarkerEndAdd = vecMarkerStartSkipped + minutes(defaultDuration);
    
        % next add these markers to the original list
        vecMarkerStartCorr = sort([vecMarkerStartCorr, vecMarkerStartSkipped]);
        vecMarkerEndCorr = sort([vecMarkerEndCorr, vecMarkerEndAdd]);
    end

end


%% Sanity check for outputs

% sanity check for equal length
if numel(vecMarkerStartCorr) ~= numel(vecMarkerEndCorr)
    error('These need to have same number of elements')
end

% sanity check for causality
for i = 1:numel(vecMarkerStartCorr)
    if ~(vecMarkerStartCorr(i) < vecMarkerEndCorr(i))
        error('Start has to be before end')
    end
end


end