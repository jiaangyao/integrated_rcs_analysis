function [vecMarkerStart, vecMarkerEnd] = strParProcPatientMarker(logTableRedactIn, ...
    boolAddStop, defaultDuration, metaDataIn)
%% strParUtils script - process patient provided disease markers
% Process the patient provided disease markers such as dyskinesia or tremor
% markers - only need to look for start/end in these cases and some times
% add start/stop post hoc
% 
% INPUT:
% logTableRedact         - table: redacted table with start and end info
% boolAddStop            - boolean: boolean for whether or not to add stop
% defaultDuration        - double: default duration of event if to be added
% metaDataIn             - struct: most general metadata struct


%% Now process the logTableRedact table

% first sort the table and create the return variables
logTableRedact = sortrows(logTableRedactIn, 'HostUnixTime');
vecMarkerStart = [];
vecMarkerEnd = [];

% then loop through all rows
for i = 1:size(logTableRedact, 1)
    % obtain relevant entries from current event
    strSubTypeCurr = logTableRedact.EventSubType{i};
    timeEventMarkerCurr = logTableRedact.('HostUnixTimeDT')(i);
    strAMorPMCurr = dtCheckAMorPM(timeEventMarkerCurr, ...
        metaDataIn.genMetaData);

    % see if current pattern is onset or offset or both
    boolOnsetCont = contains(strSubTypeCurr, 'sta', 'IgnoreCase', true) || ...
        contains(strSubTypeCurr, 'beg', 'IgnoreCase', true);
    boolOffsetCont = contains(strSubTypeCurr, 'sto', 'IgnoreCase', true) || ...
        contains(strSubTypeCurr, 'end', 'IgnoreCase', true);

    % if only onset
    if boolOnsetCont && ~boolOffsetCont
        % parse string and convert to datetime
        strRegExp = '(\s*)?\d*:\d*([aApP])?([mM])?';
        strTime = strParRegExpMarker(strSubTypeCurr, strRegExp, 1);
        timeCurr = dtStr2DateTime(strTime, metaDataIn.genMetaData, ...
            strAMorPMCurr);

        vecMarkerStart = [vecMarkerStart, timeCurr];

    % if only offset
    elseif ~boolOnsetCont && boolOffsetCont
        % parse string and convert to datetime
        strRegExp = '(\s*)?\d*:\d*([aApP])?([mM])?';
        strTime = strParRegExpMarker(strSubTypeCurr, strRegExp, 1);
        timeCurr = dtStr2DateTime(strTime, metaDataIn.genMetaData, ...
            strAMorPMCurr);

        vecMarkerEnd = [vecMarkerEnd, timeCurr];

    % if both onset and offset
    elseif boolOnsetCont && boolOffsetCont
        % first parse the start
        strRegExpStart = '[tTnN](\s*)?\d*:\d*([aApP])?([mM])?';
        strTimeStart = strParRegExpMarker(strSubTypeCurr, strRegExpStart, 2);
        timeStartCurr = dtStr2DateTime(strTimeStart, metaDataIn.genMetaData, ...
                    strAMorPMCurr);

        % now parse the stop
        strRegExpEnd = '[dDpP](\s*)?\d*:\d*([aApP])?([mM])?';
        strTimeEnd = strParRegExpMarker(strSubTypeCurr, strRegExpEnd, 2);
        timeEndCurr = dtStr2DateTime(strTimeEnd, metaDataIn.genMetaData, ...
                    strAMorPMCurr);

        % sanity check
        if ~(timeEndCurr>timeStartCurr)
            error('Start should be before end')
        end

        % now append to outer list
        vecMarkerStart = [vecMarkerStart, timeStartCurr];
        vecMarkerEnd = [vecMarkerEnd, timeEndCurr];

        % otherwise warn about the output
    else
        warning('Current unrecognized flag: %s', strSubTypeCurr)
        pause;
    end
end


%% If use specifies to add stop variables when correspondance is missing

% if user requests that missing times should be added
if boolAddStop && (numel(vecMarkerStart) > 0 | numel(vecMarkerEnd) > 0)
    [vecMarkerStart, vecMarkerEnd] = ...
        dtCorrectMarkerStartEnd(vecMarkerStart, vecMarkerEnd, ...
        'strAddorDrop', 'add', 'defaultDuration', defaultDuration);

    % enforce that sizes are equal
    assert(numel(vecMarkerStart) == numel(vecMarkerEnd), 'Size of these much be equal')
end


end