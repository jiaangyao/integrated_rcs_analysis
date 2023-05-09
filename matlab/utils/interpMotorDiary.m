function interpMD = interpMotorDiary(MD, cfg)

N_INTERP_TIME_RES = 30; % interp every 30s
N_INTERP_MINUTES_TICK = 30;

interpMD = NaN;
for i = 2:(size(MD, 1))

    if contains(cfg.str_aDBS_paradigm, 'Step5', 'IgnoreCase', true) && ...
            strcmp(cfg.str_sub, 'RCS02')
        % shift motor diary time for RCS02 step 6
        timeEndCurr = MD.time(i - 1) + minutes(15);
        if i == 2
            timeStartCurr = MD.time(i - 1);
        else
            timeStartCurr = MD.time(i - 1) - minutes(15);
        end

    elseif contains(cfg.str_aDBS_paradigm, 'Step6', 'IgnoreCase', true) && ...
            strcmp(cfg.str_sub, 'RCS02')
        timeEndCurr = MD.time(i);
        timeStartCurr = MD.time(i - 1);

    elseif contains(cfg.str_aDBS_paradigm, 'Step6', 'IgnoreCase', true) && ...
            strcmp(cfg.str_sub, 'RCS14')

        firstTimeStamp = MD.time(1);
        
        % corresponds to round 3 of Step 6 where he was filling out motor
        % diary for the next 30 min
        if firstTimeStamp.Year > 2022
            timeEndCurr = MD.time(i);
            timeStartCurr = MD.time(i - 1);
        
        % corresponds to round 1 of Step 6 where he was filling out motor
        % diary for the past 30 min
        else
            timeEndCurr = MD.time(i - 1);
            timeStartCurr = MD.time(i - 1) - minutes(30);
        end
    else
        error('Double check these values')
    end

    currEntry = MD(i - 1, :);
    currInterpFull = currEntry;

    % next figure out the total number of interp steps needed
    NTotalInterpCurr = (timeEndCurr - timeStartCurr) / seconds(N_INTERP_TIME_RES);
    for j = 1:NTotalInterpCurr
        currInterpEntry = currEntry;
        currInterpEntry.time = timeStartCurr + j * seconds(N_INTERP_TIME_RES);
        
        % avoid possible repetition of times
        if currInterpEntry.time == currEntry.time
            continue
        end
        
        currInterpEntry.MedsTaken = 0; % since want more granularity
        currInterpFull = [currInterpFull; currInterpEntry];
    end
    
    % sanity check
    if ~numel(unique(currInterpFull.time)) == size(currInterpFull, 1)
       error("No repeated time points allowed") 
    end

    % sort the rows based on times
    currInterpFull = sortrows(currInterpFull, "time");

    % next append to full table
    if i == 2
        interpMD = currInterpFull;
    else
        interpMD = [interpMD; currInterpFull];
    end
end

interpMD = [interpMD; MD(end, :)];

end