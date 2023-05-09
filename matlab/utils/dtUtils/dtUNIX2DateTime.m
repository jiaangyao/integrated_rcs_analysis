function dateTime = dtUNIX2DateTime(UNIXTimeStamp, format, timeZone)
%% dtUtils script - convert RCS UNIX times tamps to datetime format
% Converts RCS UNIX time stamps into datetime format with correct format
% and timezone information
%
% INPUT:
% UNIXTimeStamp         - double: UNIX time stamp loaded from RCS
% timeFormat            - string: format for datetime variables
% timeZone              - string: time zone for datetime variables


%% Convert time stamp into datetime format

dateTime = datetime(UNIXTimeStamp, 'ConvertFrom', 'epochtime', ...
    'TicksPerSecond', 1e3, 'Format', format, ...
    "TimeZone", "UTC");
dateTime.TimeZone = timeZone;


end