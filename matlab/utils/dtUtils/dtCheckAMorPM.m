function strAMorPM = dtCheckAMorPM(dateTimeInput, genMetaData)
%% dtUtils script - convert RCS UNIX times tamps to datetime format
% Converts RCS UNIX time stamps into datetime format with correct format
% and timezone information
%
% INPUT:
% dateTimeInput         - datetime: datetime data to be queried
% genMetaData           - struct: general metadata nested in most general
%                           metadata


%% Check if input time is in the AM or PM

% first create time for noon in the current time zone
dateTimeNoon = datetime(genMetaData.sessionYear, genMetaData.sessionMonth, ...
    genMetaData.sessionDay, 12, 00, 00, 'Format', genMetaData.timeFormat, ...
    'TimeZone', genMetaData.timeZone);

% next perform comparison
if dateTimeInput > dateTimeNoon
    strAMorPM = 'PM';
else
    strAMorPM= 'AM';
end


end