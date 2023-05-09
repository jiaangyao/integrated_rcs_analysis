function outputTable = dtAppendDateTime2Table(inputTable, ...
    vecStrUnixTime, timeFormat, timeZone, varargin)
%% dtUtils script - add readable times to settings table
% Converts datetime from unix timestamps in tables and append to original
% table as additional columns
%
% INPUT:
% timeDomainSettings    - table: time domain setting table from processRCS
% vecStrUnixTime        - cell: array of string of UNIX timestamp columns
% timeFormat            - string: format for datetime variables
% timeZone              - string: time zone for datetime variables
% 
% 
% OPTIONAL INPUT:
% boolAddStartorStop    - bool: boolean for whether to add start or stop
%                           time to the table with only one input
%                           defaul: false
% strAddStartorStop     - string: string for whether to add start or stop
%                           ['start'|'stop'] default: 'stop'
% metaDataIn            - struct: most general metadata struct
%                           default: NaN


%% Input parsing

% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

% optional commands for handling the sampling rate parsing
addParameter(p, 'boolAddStartorStop', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'strAddStartorStop', 'stop', ...
    @(x) validateattributes(x, {'char'}, {'nonempty'}));
addParameter(p, 'metaDataIn', NaN);

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
boolAddStartorStop = p.Results.boolAddStartorStop;
strAddStartorStop = p.Results.strAddStartorStop;
metaDataIn = p.Results.metaDataIn;


%% Sanity checks and preambles

% now sanity check and make sure all specified times exist in the table
for i = 1:numel(vecStrUnixTime)
    % check that all specified entries exist
    if ~any(contains(inputTable.Properties.VariableNames, ...
            vecStrUnixTime{i}, 'IgnoreCase', true))
        error("Incorrect name for specified timestamp string: %s", ...
            vecStrUnixTime{i});
    end
    
    % check that the datetime is not already inside
    if any(contains(inputTable.Properties.VariableNames, ...
            sprintf('%sDT', vecStrUnixTime{i}), 'IgnoreCase', true))
        error("Datetime name for specified timestamp string already exists: %s", ...
            vecStrUnixTime{i});
    end
    
    % check that the duration is not already inside
    if any(contains(inputTable.Properties.VariableNames, 'durationDT', ...
            'IgnoreCase', true)) && numel(vecStrUnixTime) == 2
        error("Duration in datetime already exists: %s", vecStrUnixTime{i});
    end

    % in the cases to add start or stop verify that only one column is
    % provided
    if boolAddStartorStop
        assert(numel(vecStrUnixTime) == 1, 'Only one column should be provided')
        assert(any(strcmpi(strAddStartorStop, {'start', 'stop'})), ...
            'Can only add start or stop')
        assert(isstruct(metaDataIn), 'Should provide metadata to check session start/end time')
    end
end


%% Conversion to Datetime format

% convert onset time and offset time by looping through the settings table
outputTable = table();
for i = 1:size(inputTable, 1)
    % obtain the current row and make deep copy for output
    rowCurr = inputTable(i, :);
    appendRowCurr = rowCurr(:, :);

    % convert all time entries into readable times
    for j = 1:numel(vecStrUnixTime)
        dateTime = dtUNIX2DateTime(rowCurr.(vecStrUnixTime{j}), ...
                timeFormat, timeZone);

        % handles column name based on chosen mode
        if boolAddStartorStop
            % if user specifes to have start or stop added
            if strcmpi(strAddStartorStop, 'start')
                appendRowCurr.('timeStopDT') = dateTime;
            else
                appendRowCurr.('timeStartDT') = dateTime;
            end
        else
            % default case
            appendRowCurr.(sprintf('%sDT', vecStrUnixTime{j})) = dateTime;
        end
    end

    % now also add start or stop
    % if there are only two elements and one of them contains stop and one
    % of them contains start then also compute the duration
    if numel(vecStrUnixTime) == 2
        % identify for stop and start columns
        idxStart = cellfun(@(x) contains(x, 'start', 'IgnoreCase', true), ...
            vecStrUnixTime) | cellfun(@(x) contains(x, 'onset', 'IgnoreCase', true), ...
            vecStrUnixTime);
        idxStop = cellfun(@(x) contains(x, 'stop', 'IgnoreCase', true), ...
            vecStrUnixTime) | cellfun(@(x) contains(x, 'offset', 'IgnoreCase', true), ...
            vecStrUnixTime);

        % now if we have both start and stop compute the duration
        if any(idxStart) && any(idxStop)
            strStart = vecStrUnixTime{idxStart};
            strStop = vecStrUnixTime{idxStop};

            appendRowCurr.('durationDT') = appendRowCurr.(sprintf('%sDT', strStop)) - ...
                appendRowCurr.(sprintf('%sDT', strStart));
        end

    % if otherwise user specifies to have start or stop added to table
    elseif boolAddStartorStop
        if strcmpi(strAddStartorStop, 'start')
            % if we are adding start time stamps
            % here to obtain the start session time
            if i == 1
                appendRowCurr.('timeStartDT') = metaDataIn.genMetaData.sessionStartDT;
            else
                appendRowCurr.('timeStartDT') = outputTable.timeStopDT(i - 1);
            end

            % also add in the duration
            appendRowCurr.('durationDT') = appendRowCurr.timeStopDT - ...
                appendRowCurr.timeStartDT;
        
        else
            % if we are adding stop time stamps
            if i ~= size(inputTable, 1)
                appendRowCurr.('timeStopDT') = ...
                    dtUNIX2DateTime(inputTable.(vecStrUnixTime{1})(i + 1), ...
                    timeFormat, timeZone);
            else
                appendRowCurr.('timeStopDT') = metaDataIn.genMetaData.sessionEndDT;
            end

            % also add in the duration
            appendRowCurr.('durationDT') = appendRowCurr.timeStopDT - ...
                appendRowCurr.timeStartDT;

        end
    end
    
    outputTable = [outputTable; appendRowCurr];
end


end