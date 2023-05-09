function metaDataStructOut = preprocLoadMetaDataEventMarker(...
    eventLogTable, metaDataIn)
%% Preprocessing wrapper script - loading of relevant patient self-markers
% Loads patient reported markers for start and stop of certain symptoms
% depending on subject basis
% 
% INPUT:
% eventLogTable             - table: event log table from processRCS
% metaDataIn                - struct: most general metadata struct


%% Process eventLogTable to obtain all markers with specified strings

% obtain the parsing settings for the current subject, defined via
% hardcoded rules in cfgUtils/cfgGetMarkerParseSettings
markerParseSettings = cfgGetMarkerInfo(metaDataIn);

% if enabled then perform marker parsing
if markerParseSettings.isEnabled
    % first add enabled flag
    metaDataStructOut.isEnabled = true;

    % first append datetime information to the table
    eventLogTablewTime = dtAppendDateTime2Table(eventLogTable, ...
        {'HostUnixTime'}, metaDataIn.genMetaData.timeFormat, ...
        metaDataIn.genMetaData.timeZone);
    eventLogTablewTime = dtAppendDateTime2Table(eventLogTablewTime, ...
        {'UnixOnsetTime', 'UnixOffsetTime'}, metaDataIn.genMetaData.timeFormat, ...
        metaDataIn.genMetaData.timeZone);
    
    % also subselect the entries that fit into extra comment
    warning('Verify with step 3 data to check for native report function reported markers')
    
    % extract all relevant markers from eventLogTable
    eventLogRedactFull = table();
    for i = 1:numel(markerParseSettings.vecStrKeyContain)
        eventLogTableRedactCurr = getEventwString(eventLogTablewTime, ...
            'extra_comment', markerParseSettings.vecStrKeyContain{i});
        eventLogRedactFull = [eventLogRedactFull; eventLogTableRedactCurr];
    end
    
    % next if table is non-empty then parse the table in more iteration
    if size(eventLogRedactFull, 1) ~= 0
        [metaDataStructOut.vecMarkerStart, metaDataStructOut.vecMarkerEnd] = ...
            strParProcPatientMarker(eventLogRedactFull, ...
            markerParseSettings.boolAddStop, ...
            markerParseSettings.defaultDuration, metaDataIn);
    
        % wait for user check in case script fails to process some data
        if size(eventLogRedactFull, 1) > ...
                max([numel(metaDataStructOut.vecMarkerStart), ...
                numel(metaDataStructOut.vecMarkerEnd)])
            pause;
        end
    
    % otherwise add empty arrays
    else
        metaDataStructOut.vecMarkerStart = [];
        metaDataStructOut.vecMarkerEnd = [];
    end

% if not enabled then append not enabled flag
else
    metaDataStructOut.isEnabled = false;
end


%% Also backup the original metadata

% add the table with datetime added to the output structure
metaDataStructOut.origMetaData = eventLogTablewTime;

% also add the setting struct to output
metaDataStructOut.markerParseSettings = markerParseSettings;


end