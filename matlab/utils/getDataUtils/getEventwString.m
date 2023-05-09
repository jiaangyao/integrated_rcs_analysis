function logTableOut = getEventwString(logTable, strEventType, ...
    strKeyContain)
%% getDataUtils script - loading of all event containing specific strings
% Loads channel name for all recording channels in the time domain based
% off timeDomainSettings
% 
% INPUT:
% logTable          - table: time domain setting table from processRCS
% strEventType      - string: which event type to search for
% strKeyContain     - string: string to be searched for in EventSubType


%% Search through all rows of the input table

% create empty table and iterate through all rows
logTableOut = table();
for i = 1:size(logTable, 1)
    rowCurr = logTable(i, :);
    
    % first test for event type
    if strcmpi(rowCurr.EventType, strEventType) || ...
            contains(rowCurr.EventType, strEventType, "IgnoreCase", true)
        % next search for query string
        if contains(rowCurr.EventSubType, strKeyContain, ...
            "IgnoreCase", true)
            logTableOut = [logTableOut, rowCurr];
        end
    end
end


end
