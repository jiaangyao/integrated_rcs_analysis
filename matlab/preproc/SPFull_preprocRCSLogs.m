function output = preprocessRCSLogs(p_data, str_session, str_device_curr, cfg)
%% obtain the path and call ProcessRCS

p_path_in  = fullfile(p_data, str_session, str_device_curr);

% sanity check
if exist(p_path_in, 'dir') ~= 7
    error("The input directory should exist")
end

fprintf('Loading %s...\n', p_path_in)

% run ProcessRCS
[unifiedDerivedTimes,...
    timeDomainData, timeDomainData_onlyTimeVariables, timeDomain_timeVariableNames,...
    AccelData, AccelData_onlyTimeVariables, Accel_timeVariableNames,...
    PowerData, PowerData_onlyTimeVariables, Power_timeVariableNames,...
    FFTData, FFTData_onlyTimeVariables, FFT_timeVariableNames,...
    AdaptiveData, AdaptiveData_onlyTimeVariables, Adaptive_timeVariableNames,...
    timeDomainSettings, powerSettings, fftSettings, eventLogTable,...
    metaData, stimSettingsOut, stimMetaData, stimLogSettings,...
    DetectorSettings, AdaptiveStimSettings, AdaptiveEmbeddedRuns_StimSettings,...
    versionInfo] = ProcessRCS(p_path_in, 3);


%% Now load important meta data

fs = timeDomainSettings.samplingRate;
if numel(fs) > 1
    if ~all(fs == fs(1))
        error('should all be equal')
    else
        fs = fs(1);
    end
end

% obtain the string of all channels
vec_str_chan = {};
vec_str_var_chan = {};
for i = 1:8
    str_chan_curr = sprintf("chan%d", i);
    if any(strcmp(timeDomainSettings.Properties.VariableNames, str_chan_curr)) && ...
            i >= 5
        error("More than four channels in this file?")
    end

    if any(strcmp(timeDomainSettings.Properties.VariableNames, str_chan_curr))
        str_chan_full_curr = split(timeDomainSettings.(str_chan_curr));
        str_chan_loc = str_chan_full_curr{1};
        vec_str_chan{i} = str_chan_loc;
        vec_str_var_chan{i} = str_chan_curr;
    end

end

% create output structure for later
output.fs = fs;
output.vec_str_chan = vec_str_chan;

% obtain absolute time of recording start and end
output.metaData.sessionStart = timeDomainData.localTime(1);
output.metaData.sessionEnd = timeDomainData.localTime(end);

output.metaData.sessionYear = output.metaData.sessionStart.Year;
output.metaData.sessionMonth = output.metaData.sessionStart.Month;
output.metaData.sessionDay = output.metaData.sessionStart.Day;
output.metaData.timeZone = output.metaData.sessionStart.TimeZone;
output.metaData.timeFormat = output.metaData.sessionStart.Format;

% obtain relevant metadata for adaptive
strCurrSide = getStrSideFromDeviceName(str_device_curr, cfg);
output = loadAdaptiveMetaData(AdaptiveEmbeddedRuns_StimSettings, ...
    DetectorSettings, powerSettings, vec_str_chan, output, strCurrSide, cfg);

% load patient self-reported markers
output = preprocLoadEventMarker(eventLogTable, output, cfg);

%% load the logs

% form the path for globbing
strLogFolder = sprintf('LogDataFrom%s*', cfg.vec_str_side{1});
vecPathLogFiles = glob(fullfile(p_path_in, strLogFolder, '*AppLog*'));
if length(vecPathLogFiles) ~= 1
    error('Should be one log file exactly')
end

% obtain the lags
lag = getINSLags(cfg.str_sub);

% define the import options
opts = detectImportOptions(vecPathLogFiles{1});
opts = setvartype(opts, "Var2", 'char');

% load the log file
logTable = readtable(vecPathLogFiles{1}, opts);

% obtain the start and end datetime
startDT = sprintf("%d/%d/%d 00:00:01 AM", output.metaData.sessionMonth, ...
    output.metaData.sessionDay, output.metaData.sessionYear);
endDT = sprintf("%d/%d/%d 11:59:59 PM", output.metaData.sessionMonth, ...
    output.metaData.sessionDay, output.metaData.sessionYear);

%% extract new state and time

% loop through all entries and try to find current
count = 0;
newStateLogInd = [];
vecNewState = [];
vecOldState = [];
vecNewTime = [];
vecNewAmp = [];

% figure out if the variables are cell or arrays
if iscell(logTable.Var2)
    boolVar2Cell = true;
else
    boolVar2Cell = false;
    warning('Var2 is being loaded as an array')
end

if iscell(logTable.Var3)
    boolVar3Cell = true;
    warning("Var3 is being loaded as a cell array")
else
    boolVar3Cell = false;
end

for i = 1:length(logTable.LogEntry_Header)
    temp = contains('AdaptiveTherapyModificationEntry.NewState',logTable.LogEntry_Header(i));
    if temp == 1
        % get time of current entry
        currTime = logTable.Var3(i - 5);

        % correct for lag in INS if defined for current subject
        if lag ~= 0
            currTime = currTime - lag;
        end

        if currTime >= startDT & currTime <= endDT

            % append the indices
            newStateLogInd = [newStateLogInd, i];
            count = count + 1;

    
            % append the new state
            if boolVar2Cell
                newState = hex2dec(logTable.Var2{i});
            else
                newState = logTable.Var2(i);
            end
            vecNewState = [vecNewState; newState];

            % append the old state
            if boolVar2Cell
                oldState = hex2dec(logTable.Var2{i + 1});
            else
                oldState = logTable.Var2(i + 1);
            end
            vecOldState = [vecOldState; oldState];
    
            % append the time
            if boolVar3Cell
                newTime = logTable.Var3{i - 5};
            else
                newTime = logTable.Var3(i - 5);
            end
            vecNewTime = [vecNewTime; newTime];

            % append the amplitude
            if boolVar2Cell
                newAmp = str2num(logTable.Var2{i + 7});
            else
                newAmp = logTable.Var2(i + 7);
            end
            vecNewAmp = [vecNewAmp; newAmp];
        end
    end
end

% % get rid of the last entry
% vecNewState = vecNewState(1: end-1);
% vecNewTime = vecTime(1:end - 1);

%% now form the state and amplitude variable for plotting

timeVector = (vecNewTime(end):seconds(1):vecNewTime(1)).';
stateVector = zeros(length(timeVector),1);
ampVector = zeros(length(timeVector),1);
for i = 2:length(vecNewState)
    % form the state vector
    ind = timeVector >= vecNewTime(i) & timeVector < vecNewTime(i - 1);
    stateVector(ind) = vecNewState(i);
    assert(vecOldState(i - 1) == vecNewState(i), 'Old state new state mismatch')

    % form the amplitude vector
    ampVector(ind) = vecNewAmp(i - 1);
end
timeVector.TimeZone = output.metaData.timeZone;

% compute the aliased state vector
aliasedStateVector = stateVector;
aliasedStateVector(stateVector == 3) = 0;
aliasedStateVector(stateVector == 4) = 1;

% form the output structure 
logData.time = timeVector;
logData.state = stateVector;
logData.amp = ampVector;
logData.aliasedState = aliasedStateVector;

%% return final output

output.logData = logData;

end