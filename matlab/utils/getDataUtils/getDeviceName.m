function strDeviceID = getDeviceName(strSubject, strSide)
%% getDataUtils script - loading of Device ID for RCS patients
% Loads Device ID based on subject name and which side the recording is on
% Note: all of the Device IDs in this script are hardcoded so make sure to
% very them
% 
% INPUT:
% strSubject    - string: string for the name of the subject, e.g. RCS01
% strSide       - string: string for the side, L/R


%% warning and sanity check
% warning('Note: Device IDs are hardcoded right now and make sure to verify them')
if ~(strcmpi(strSide, 'L') || strcmpi(strSide, 'R'))
    error('Side of lead implant can only be L or R, current input: %s', strSide)
end


%% actual comparison and querying for device ID

if strcmp(strSubject, "RCS02")
    if strcmp(strSide, 'L')
        strDeviceID = 'DeviceNPC700398H';
    else
        strDeviceID = 'DeviceNPC700404H';
    end

elseif strcmp(strSubject, "RCS08")
    if strcmp(strSide, 'L')
        strDeviceID = 'DeviceNPC700444H';
    else
        strDeviceID = 'DeviceNPC700421H';
    end

elseif strcmp(strSubject, 'RCS14')
    if strcmp(strSide, 'L')
        strDeviceID = 'DeviceNPC700481H';
    else
        error('Unilateral implant')
    end

elseif strcmp(strSubject, 'RCS17')
    if strcmp(strSide, 'L')
        strDeviceID = 'DeviceNPC700545H';
    else
        error('NotImplementedError')
    end

else
    error("Device ID not defined for subject: %s", strSubject)
end


end