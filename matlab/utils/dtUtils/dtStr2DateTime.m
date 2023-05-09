function timeCurr = dtStr2DateTime(strTime, genMetaData, strAMorPM)
%% dtUtils script - convert string time to datetime format
% Converts str times into datetime format with correct format
% and timezone information
%
% INPUT:
% UNIXTimeStamp         - double: UNIX time stamp loaded from RCS
% genMetaData           - struct: general metadata nested in most general
%                           metadata
% strAMorPM             - string: whether the current time is AM or PM
%                           ['AM'|'PM']

%% Convert string time to datetime format

% first check with the time
timeCurr = datetime(datestr(strTime), 'InputFormat', ...
    'dd-MMM-yyyy HH:mm:SS');
timeCurr.Year = genMetaData.sessionYear;
timeCurr.Month = genMetaData.sessionMonth;
timeCurr.Day = genMetaData.sessionDay;
timeCurr.Format = genMetaData.timeFormat;
timeCurr.TimeZone = genMetaData.timeZone;

% compare with input AM or PM flag to determine if offset is needed
assert(any(strcmpi(strAMorPM, {'AM', 'PM'})), 'Incorrect AM/PM flag provided')
if ~strcmpi(strAMorPM, dtCheckAMorPM(timeCurr, genMetaData))
    if strcmpi(strAMorPM, 'AM')
        timeCurr = timeCurr - hours(12);
    else
        timeCurr = timeCurr + hours(12);
    end
end


end