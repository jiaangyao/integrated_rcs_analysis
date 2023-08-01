function plotFeaturesOverTime(pFigure, vec_output, AppleWatchTable, pkgWatchTable, ...
    motorDiaryInterp, motorDiary, str_on_off_meds, str_side, str_data_day, cfg_in, ...
    varargin)
%% Input parsing
% adding custom input flag so can implement new path searching without
% modifying existing scripts - JY 07/18/2022

% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'boolSaveAsFig', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
boolSaveAsFig = p.Results.boolSaveAsFig;

% Define all magic numbers
N_SMOOTHING_FILTER_LENGTH_SEC = 5;

%%

cfg = cfg_in;

% loop through all the sessions
t_raw_data_full = [];
raw_data_full = [];

pow_data_stn_full = [];
pow_data_motor_full = [];

t_pow_abs_stn_full = [];
t_pow_abs_motor_full = [];

t_adap_abs_ld0_full = [];
t_adap_abs_ld1_full = [];

ld0_data_full = [];
ld1_data_full = [];

t_adap_full = [];
current_in_full = [];
state_full = [];

vecDyskOnsetFull = [];
vecDyskOffsetFull = [];

vecLDCombo = [];

for idx_sess = 1:numel(vec_output)
    % unpack the variables
    output_curr = vec_output{idx_sess};

    t_raw_data_curr = output_curr.time_abs;
    raw_data_curr = output_curr.raw_data;
    
    pow_data_motor_curr = output_curr.pow_data_motor;
    % pow_data_stn_curr = output_curr.pow_data_stn_wnan;
    pow_data_stn_curr = output_curr.pow_data_stn;

    % t_pow_abs_stn_curr = output_curr.t_pow_abs_wnan;
    t_pow_abs_stn_curr = output_curr.t_pow_abs;
    t_pow_abs_motor_curr = output_curr.t_pow_abs;

    % t_adap_abs_ld0_curr = output_curr.t_adap_abs_wnan;
    t_adap_abs_ld0_curr = output_curr.t_adap_abs;
    t_adap_abs_ld1_curr = output_curr.t_adap_abs;

    % ld0_data_curr = output_curr.ld0_data_wnan;
    ld0_data_curr = output_curr.ld0_data;
    ld1_data_curr = output_curr.ld1_data;
    
    t_adap_curr = output_curr.t_adap_abs;
    current_in_curr = output_curr.vec_currrent_in(:, 1);
    
    state_curr = output_curr.vec_idx_state;
    
    % also obtain the patient defined dysk onset and offsets
    vecDyskOnsetCurr = output_curr.vecDyskOnset;
    vecDyskOffsetCurr = output_curr.vecDyskOffset;

    % now pack to outer structure
    t_raw_data_full = [t_raw_data_full; t_raw_data_curr];
    raw_data_full = [raw_data_full; raw_data_curr];

    pow_data_stn_full = [pow_data_stn_full; pow_data_stn_curr];
    pow_data_motor_full = [pow_data_motor_full; pow_data_motor_curr];

    t_pow_abs_stn_full = [t_pow_abs_stn_full; t_pow_abs_stn_curr];
    t_pow_abs_motor_full = [t_pow_abs_motor_full; t_pow_abs_motor_curr];
    
    % append everything regardless
    t_adap_abs_ld0_full = [t_adap_abs_ld0_full; t_adap_abs_ld0_curr];
    t_adap_abs_ld1_full = [t_adap_abs_ld1_full; t_adap_abs_ld1_curr];

    ld0_data_full = [ld0_data_full; ld0_data_curr];
    ld1_data_full = [ld1_data_full; ld1_data_curr];

    t_adap_full = [t_adap_full; t_adap_curr];
    current_in_full = [current_in_full; current_in_curr];
    state_full = [state_full; state_curr];

    vecDyskOnsetFull = [vecDyskOnsetFull, vecDyskOnsetCurr];
    vecDyskOffsetFull = [vecDyskOffsetFull, vecDyskOffsetCurr];

    % also optionally append LDCombo in case combo threshold is being used
    if cfg.thresCombo
        vecLDCombo = [vecLDCombo; output_curr.LDCombo];
        weightVector = output_curr.weightVector;
    end

end

% first obtain the valid time stamps from the STN power array
% and also sort the array
idx_valid_pow_stn_full = ~isnat(t_pow_abs_stn_full);
t_pow_abs_stn_valid_full = t_pow_abs_stn_full(idx_valid_pow_stn_full);
pow_data_stn_valid_full = pow_data_stn_full(idx_valid_pow_stn_full, :);

[~, idx_arg_sort_pow_stn] = sort(t_pow_abs_stn_valid_full);
t_pow_abs_stn_valid_sorted = t_pow_abs_stn_valid_full(idx_arg_sort_pow_stn);
pow_data_stn_valid_sorted = pow_data_stn_valid_full(idx_arg_sort_pow_stn, :);

% next sort the motor power data
[~, idx_arg_sort_pow_motor] = sort(t_pow_abs_motor_full);
t_pow_abs_motor_sorted = t_pow_abs_motor_full(idx_arg_sort_pow_motor);
pow_data_motor_sorted = pow_data_motor_full(idx_arg_sort_pow_motor, :);

% subsequently sort the LD0 data
idx_valid_ld0_full = ~isnat(t_adap_abs_ld0_full);
t_adap_abs_ld0_valid_full = t_adap_abs_ld0_full(idx_valid_ld0_full);
ld0_data_valid_full = ld0_data_full(idx_valid_ld0_full, :);

[~, idx_arg_sort_ld0] = sort(t_adap_abs_ld0_valid_full);
t_adap_abs_ld0_valid_sorted = t_adap_abs_ld0_valid_full(idx_arg_sort_ld0);
ld0_data_valid_sorted = ld0_data_valid_full(idx_arg_sort_ld0, :);

% next also sort the LD1 data
idx_valid_ld1_full = ~isnat(t_adap_abs_ld1_full);
t_adap_abs_ld1_valid_full = t_adap_abs_ld1_full(idx_valid_ld1_full);
ld1_data_valid_full = ld1_data_full(idx_valid_ld1_full, :);

[~, idx_arg_sort_ld1] = sort(t_adap_abs_ld1_valid_full);
t_adap_abs_ld1_valid_sorted = t_adap_abs_ld1_valid_full(idx_arg_sort_ld1);
ld1_data_valid_sorted = ld1_data_valid_full(idx_arg_sort_ld1, :);

% finally sort the current and state variables
idx_state_valid = ~isnan(state_full);
t_adap_valid = t_adap_full(idx_state_valid);
state_valid = state_full(idx_state_valid);
current_in_valid = current_in_full(idx_state_valid);

[~, idx_arg_sort_adap] = sort(t_adap_valid);
t_adap_valid_sorted = t_adap_valid(idx_arg_sort_adap);
current_in_valid_sorted = current_in_valid(idx_arg_sort_adap, :);
state_valid_sorted = state_valid(idx_arg_sort_adap, :);

% obtain all metadata
LD0_adaptiveMetaData = ....
    vec_output{1}.adaptiveMetaData.LD0.(str_side(1));
LD1_adaptiveMetaData = ....
    vec_output{1}.adaptiveMetaData.LD1.(str_side(1));

% account for fake adaptive settings

% optionally sort the combined LD in cases where it's present
if cfg.thresCombo
    vecLDComboValidFull = vecLDCombo(idx_valid_ld0_full);
    vecLDComboValidSorted = vecLDComboValidFull(idx_arg_sort_ld0);
end


% collapse some of the states together
if ~any(strcmp(cfg.str_data_day, {'20221205'}))
    % now if only driven by LD0
    state_valid_sorted_remove34 = state_valid_sorted;
    state_valid_sorted_remove34(state_valid_sorted == 3) = 0;
    state_valid_sorted_remove34(state_valid_sorted == 4) = 1;
    
    % now if only driven by LD1
    state_valid_sorted_remove14 = state_valid_sorted;
    state_valid_sorted_remove14(state_valid_sorted == 1) = 0;
    state_valid_sorted_remove14(state_valid_sorted == 4) = 3;

    highStimLevel = LD0_adaptiveMetaData.stimLevel(end);
    lowStimLevel = LD0_adaptiveMetaData.stimLevel(1);
    
    % collapse some of the current
    if strcmp(cfg.str_sub, 'RCS17')
        if ~LD0_adaptiveMetaData.boolReverse || ...
                ~LD1_adaptiveMetaData.boolReverse
            warning('Enforcing Gamma Biomaker reverse')
            LD0_adaptiveMetaData.boolReverse = true;
            LD1_adaptiveMetaData.boolReverse = true;
        end

        % correct for stim level the same
        if highStimLevel == lowStimLevel
            if strcmp(cfg.vec_str_side{1}, 'Left')
                highStimLevel = 3.1;
                lowStimLevel = 2.3;
            else
                highStimLevel = 2.4;
                lowStimLevel = 1.8;
            end
        end
    end
    % now if only driven by LD0
    current_in_valid_sorted_remove34 = NaN(size(state_valid_sorted_remove34));
    if ~LD0_adaptiveMetaData.boolReverse
        current_in_valid_sorted_remove34(state_valid_sorted_remove34 == 0) = lowStimLevel;
        current_in_valid_sorted_remove34(state_valid_sorted_remove34 == 1) = highStimLevel;
    else
        current_in_valid_sorted_remove34(state_valid_sorted_remove34 == 0) = highStimLevel;
        current_in_valid_sorted_remove34(state_valid_sorted_remove34 == 1) = lowStimLevel;
    end

    % now if only driven by LD1
    current_in_valid_sorted_remove14 = NaN(size(state_valid_sorted_remove14));
    if ~LD1_adaptiveMetaData.boolReverse
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 0) = lowStimLevel;
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 3) = highStimLevel;
    else
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 3) = lowStimLevel;
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 0) = highStimLevel;
    end

else
    % now if only driven by LD0
    state_valid_sorted_remove34 = state_valid_sorted;
    state_valid_sorted_remove34(state_valid_sorted == 3) = 0;
    state_valid_sorted_remove34(state_valid_sorted == 6) = 0;
    state_valid_sorted_remove34(state_valid_sorted == 4) = 1;
    state_valid_sorted_remove34(state_valid_sorted == 7) = 1;
    
    % now if only driven by LD1
    state_valid_sorted_remove14 = state_valid_sorted;
    state_valid_sorted_remove14(state_valid_sorted == 1) = 0;
    state_valid_sorted_remove14(state_valid_sorted == 4) = 3;
    state_valid_sorted_remove14(state_valid_sorted == 7) = 6;
    
    % collapse some of the current
    % now if only driven by LD0
    current_in_valid_sorted_remove34 = NaN(size(state_valid_sorted_remove34));
    current_in_valid_sorted_remove34(state_valid_sorted_remove34 == 0) = output_curr.lowStimLevel;
    current_in_valid_sorted_remove34(state_valid_sorted_remove34 == 1) = output_curr.highStimLevel;
    
    highStimLevel = output_curr.highStimLevel;
    lowStimLevel = output_curr.lowStimLevel;
    if strcmp(str_side, 'Left')
        midStimLevel = 2.4;
    else
        midStimLevel = 3.1;
    end

    % now if only driven by LD1
    current_in_valid_sorted_remove14 = NaN(size(state_valid_sorted_remove14));
    if strcmp(cfg.str_round, 'fifth round') && strcmp(str_side, 'Left')
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 0) = highStimLevel;
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 3) = midStimLevel;
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 6) = lowStimLevel;

    
    elseif strcmp(cfg.str_round, 'fifth round') && strcmp(str_side, 'Right')
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 0) = highStimLevel;
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 3) = midStimLevel;
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 6) = lowStimLevel;
    else
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 0) = lowStimLevel;
        current_in_valid_sorted_remove14(state_valid_sorted_remove14 == 3) = highStimLevel;
    end
end


%% also extract information from the motor diary

% extract valid entries from motor diary
if ~any(strcmp(cfg.str_no_md_data_day, cfg.str_data_day))
    idxMotorDiaryValid = motorDiary.time >= t_adap_valid_sorted(1) & ...
        motorDiary.time <= (t_adap_valid_sorted(end) + minutes(30));
    motorDiaryRedacted = motorDiary(idxMotorDiaryValid, :);
    
    % TODO: change this into a function
    % correct for RCS02 motor diary during plotting - note lazy way of dealing
    % with things
    if contains(cfg.str_aDBS_paradigm, 'Step5', 'IgnoreCase', true) && ...
                strcmp(cfg.str_sub, 'RCS02')
        motorDiaryRedacted.time = motorDiaryRedacted.time + minutes(15);
    
    elseif contains(cfg.str_aDBS_paradigm, 'Step6', 'IgnoreCase', true) && ...
                strcmp(cfg.str_sub, 'RCS02')
        motorDiaryRedacted.time = motorDiaryRedacted.time + minutes(30);
    elseif contains(cfg.str_aDBS_paradigm, 'Step6', 'IgnoreCase', true) && ...
                strcmp(cfg.str_sub, 'RCS14')
        motorDiaryRedacted.time = motorDiaryRedacted.time + minutes(30);
    end
    
    % extract valid entries from interpolated motor diary
    idxMotorDiaryInterpValid = motorDiaryInterp.time >= t_adap_valid_sorted(1) & ...
        motorDiaryInterp.time <= t_adap_valid_sorted(end);
    motorDiaryInterpRedacted = motorDiaryInterp(idxMotorDiaryInterpValid, :);
    
    % figure out all medication time
    idxMedTime = motorDiaryInterp.MedsTaken == 1;
    medTime = motorDiaryInterp.time(idxMedTime);

else
    motorDiaryRedacted = NaN;
    motorDiaryInterpRedacted = NaN;
    medTime = NaN;
end


% optionally process the Apple Watch data
if ~any(strcmp(cfg.str_no_aw_data_day, cfg.str_data_day))
    % finally finally obtain the apple watch data
    t_dys_full = AppleWatchTable.t_dys_full;
    prob_dys_full = AppleWatchTable.prob_dys_full;
    
    idxDysValid = t_dys_full >= t_adap_valid_sorted(1) & ...
        t_dys_full <= t_adap_valid_sorted(end);
    t_dys_full = t_dys_full(idxDysValid);
    prob_dys_full = prob_dys_full(idxDysValid);
    
    t_tre_full = AppleWatchTable.t_tre_full;
    prob_tre_full = AppleWatchTable.prob_tre_full;
    
    idxTreValid = t_tre_full >= t_adap_valid_sorted(1) & ...
        t_tre_full <= t_adap_valid_sorted(end);
    t_tre_full = t_tre_full(idxTreValid);
    prob_tre_full = prob_tre_full(idxTreValid);

    if strcmp(cfg.str_sub, 'RCS14')
        t_accel_full = AppleWatchTable.t_accel_full;
        accel_full = AppleWatchTable.accel_full;

        idxAccelValid = t_accel_full >= t_adap_valid_sorted(1) & ...
            t_accel_full <= t_adap_valid_sorted(end);
        t_accel_full = t_accel_full(idxAccelValid);
        accel_full = accel_full(idxAccelValid);
    end
end

% optionally process the PKG Watch data
if ~any(strcmp(cfg.str_no_pkg_data_day, cfg.str_data_day))
    % first obtain all valid entries from the PKG table
    boolPKGValid = (~[pkgWatchTable.Off_Wrist{:}])' & ~isnan(pkgWatchTable.BK) & ...
        ~isnan(pkgWatchTable.DK); 
    pkgWatchTableRedacted = pkgWatchTable(boolPKGValid, ...
        {'Date_Time', 'BK', 'DK', 'Off_Wrist'});
    
    % also limit to current day
    idxPKGDateValid = pkgWatchTableRedacted.Date_Time >= t_pow_abs_stn_valid_sorted(1) & ...
        pkgWatchTableRedacted.Date_Time <= t_pow_abs_stn_valid_sorted(end);
    pkgWatchTableRedacted = pkgWatchTableRedacted(idxPKGDateValid, :);

    % next remove 0 in dysk scores
    idxDyskZero = pkgWatchTableRedacted.DK == 0;
    pkgWatchTableRedacted.DK(idxDyskZero) = ...
        pkgWatchTableRedacted.DK(idxDyskZero) + 0.1;
end

%% some hacky ways to adjust the ways things are named for RCS02

if strcmp(cfg.str_sub, 'RCS02')
    if strcmp(cfg.str_data_day, '20220914')
        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timeRaw = t_raw_data_full;
        rawData = raw_data_full;

        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted;

        timeAWDysk = t_dys_full; AWDyskData = prob_dys_full;
        timePKG = pkgWatchTableRedacted.Date_Time;
        PKGBradyData = pkgWatchTableRedacted.BK;
        PKGDyskData = pkgWatchTableRedacted.DK;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = true; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.8, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        % note that LD1 data from this day had some issues so have to hard code
        % to remove portion before around 10:20AM
        boolEnabled_LD1 = true;
        idxLDDataValid = ld1_data_valid_sorted(:, 1) ~= 0;
        timeLD_LD1_Valid = t_adap_abs_ld1_valid_sorted(idxLDDataValid);
        LDData_LD1_Valid = ld1_data_valid_sorted(idxLDDataValid, 1);

        % also correct for adaptive state data
        idxStateValid = t_adap_valid_sorted >= timeLD_LD1_Valid(1) & ...
            t_adap_valid_sorted <= timeLD_LD1_Valid(end);
        timeStateLD1_Valid = t_adap_valid_sorted(idxStateValid);
        stateDataLD1_Valid = state_valid_sorted_remove14(idxStateValid);
        currentDataLD1_Valid = current_in_valid_sorted_remove14(idxStateValid);

        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = timeLD_LD1_Valid;
        LDData_LD1 = LDData_LD1_Valid;
        LDThresh_LD1 = 35;

        timeStateLD1 = timeStateLD1_Valid;
        stateDataLD1 = stateDataLD1_Valid;
        currentDataLD1 = currentDataLD1_Valid;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.8, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    elseif strcmp(cfg.str_data_day, '20220916')
        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        timeAWDysk = t_dys_full; AWDyskData = prob_dys_full;
        timePKG = pkgWatchTableRedacted.Date_Time;
        PKGBradyData = pkgWatchTableRedacted.BK;
        PKGDyskData = pkgWatchTableRedacted.DK;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.8, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.8, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    elseif strcmp(cfg.str_data_day, '20220918')
        % combo day
        % first obtain both power bands
        timeResCxtWin = 2;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        % preprocess the LD combo data
        vecLDComboValidSortedOrig = vecLDComboValidSorted;
        vecLDComboValidSorted(vecLDComboValidSorted > 2^31) = ...
            vecLDComboValidSorted(vecLDComboValidSorted > 2^31) - 2^32;

        % next obtain the LD features
        timeLD_LDCombo = t_adap_abs_ld0_valid_sorted;
        LDData_LD0Input = ld0_data_valid_sorted(:, 1);
        LDData_LD1Input = ld0_data_valid_sorted(:, 2);
        LDData_LDCombo = vecLDComboValidSorted;
        LDThresh_LDCombo = ld0_thres_low;

        % finally obtain the state and current information
        timeStateLDCombo = t_adap_valid_sorted;
        stateDataLDCombo = state_valid_sorted_remove34;
        currentDataLDCombo = current_in_valid_sorted;

        timeAWDysk = t_dys_full; AWDyskData = prob_dys_full;
        timePKG = pkgWatchTableRedacted.Date_Time;
        PKGBradyData = pkgWatchTableRedacted.BK;
        PKGDyskData = pkgWatchTableRedacted.DK;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        % define the string parameters
        strLD_LDCombo = 'LD Combo'; strLD_LD0 = 'LD0 Input'; strLD_LD1 = 'LD1 Input';
        strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

        colorLDCombo = [0, 0.4470, 0.7410]; smoothColorLDCombo = [1, 0, 0];
        colorLD0 = [0, 0.4470, 0.7410]; colorLD1 = [0.8500, 0.3250, 0.0980];
        ylimLD_LDCombo = [-300, 2000];
        ylimPowLD0 = [0, 2000]; ylimLD_LD0 = [0, 500];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100];
        ylimState_LDCombo = [-0.5, 1.5]; ylimCurrent_LDCombo = [2.8, 3.6];
        ylimMDRange_LDCombo = [-0.5, 3.5];


    elseif strcmp(cfg.str_data_day, '20220920')
        % note that because this day was driven by gamma which is placed at
        % LD0, state_remove34 actually corresponds to gamma
        state_valid_sorted_remove34_corr = state_valid_sorted_remove14;
        state_valid_sorted_remove34_corr(state_valid_sorted_remove14 == 3) = 1;

        current_in_valid_sorted_remove34_corr = ...
            NaN(size(state_valid_sorted_remove34_corr));
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 0) = 3.0;
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 1) = highStimLevel;

        state_valid_sorted_remove14_corr = state_valid_sorted_remove34;
        state_valid_sorted_remove14_corr(state_valid_sorted_remove34 == 1) = 3;
        current_in_valid_sorted_remove14_corr = ...
            current_in_valid_sorted_remove34;

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld1_valid_sorted;
        LDData_LD0 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld1_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34_corr;
        currentDataLD0 = current_in_valid_sorted_remove34_corr;

        timeAWDysk = t_dys_full; AWDyskData = prob_dys_full;
        timePKG = pkgWatchTableRedacted.Date_Time;
        PKGBradyData = pkgWatchTableRedacted.BK;
        PKGDyskData = pkgWatchTableRedacted.DK;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = true; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.8, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld0_valid_sorted;
        LDData_LD1 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld0_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14_corr;
        currentDataLD1 = current_in_valid_sorted;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.8, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    elseif strcmp(cfg.str_data_day, '20220922')
        % combo day in second round
        % first obtain both power bands
        timeResCxtWin = 2;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        % preprocess the LD combo data
        vecLDComboValidSortedOrig = vecLDComboValidSorted;
        vecLDComboValidSorted(vecLDComboValidSorted > 2^31) = ...
            vecLDComboValidSorted(vecLDComboValidSorted > 2^31) - 2^32;

        % next obtain the LD features
        timeLD_LDCombo = t_adap_abs_ld0_valid_sorted;
        LDData_LD0Input = ld0_data_valid_sorted(:, 1);
        LDData_LD1Input = ld0_data_valid_sorted(:, 2);
        LDData_LDCombo = vecLDComboValidSorted;
        LDThresh_LDCombo = ld0_thres_low;

        % finally obtain the state and current information
        timeStateLDCombo = t_adap_valid_sorted;
        stateDataLDCombo = state_valid_sorted_remove34;
        currentDataLDCombo = current_in_valid_sorted;

        timeAWDysk = t_dys_full; AWDyskData = prob_dys_full;
        timePKG = pkgWatchTableRedacted.Date_Time;
        PKGBradyData = pkgWatchTableRedacted.BK;
        PKGDyskData = pkgWatchTableRedacted.DK;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        % define the string parameters
        strLD_LDCombo = 'LD Combo'; strLD_LD0 = 'LD0 Input'; strLD_LD1 = 'LD1 Input';
        strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

        colorLDCombo = [0, 0.4470, 0.7410]; smoothColorLDCombo = [1, 0, 0];
        colorLD0 = [0, 0.4470, 0.7410]; colorLD1 = [0.8500, 0.3250, 0.0980];
        ylimLD_LDCombo = [-300, 2000];
        ylimPowLD0 = [0, 2000]; ylimLD_LD0 = [0, 500];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100];
        ylimState_LDCombo = [-0.5, 1.5]; ylimCurrent_LDCombo = [2.8, 3.6];
        ylimMDRange_LDCombo = [-0.5, 4.5];

    elseif strcmp(cfg.str_data_day, '20220925')
        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = pkgWatchTableRedacted.Date_Time;
        PKGBradyData = pkgWatchTableRedacted.BK;
        PKGDyskData = pkgWatchTableRedacted.DK;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = true; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.8, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.8, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    elseif strcmp(cfg.str_data_day, '20220927')
        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        timeAWDysk = t_dys_full; AWDyskData = prob_dys_full;
        timePKG = pkgWatchTableRedacted.Date_Time;
        PKGBradyData = pkgWatchTableRedacted.BK;
        PKGDyskData = pkgWatchTableRedacted.DK;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.8, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.8, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    elseif strcmp(cfg.str_data_day, '20220928')
        % note that because this day was driven by gamma which is placed at
        % LD0, state_remove34 actually corresponds to gamma
        state_valid_sorted_remove34_corr = state_valid_sorted_remove14;
        state_valid_sorted_remove34_corr(state_valid_sorted_remove14 == 3) = 1;

        current_in_valid_sorted_remove34_corr = ...
            NaN(size(state_valid_sorted_remove34_corr));
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 0) = lowStimLevel;
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 1) = highStimLevel;

        state_valid_sorted_remove14_corr = state_valid_sorted_remove34;
        state_valid_sorted_remove14_corr(state_valid_sorted_remove34 == 1) = 3;
        current_in_valid_sorted_remove14_corr = ...
            current_in_valid_sorted_remove34;

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld1_valid_sorted;
        LDData_LD0 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld1_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34_corr;
        currentDataLD0 = current_in_valid_sorted_remove34_corr;

        timeAWDysk = t_dys_full; AWDyskData = prob_dys_full;
        timePKG = pkgWatchTableRedacted.Date_Time;
        PKGBradyData = pkgWatchTableRedacted.BK;
        PKGDyskData = pkgWatchTableRedacted.DK;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = true; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.8, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1  = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld0_valid_sorted;
        LDData_LD1 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld0_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14_corr;
        currentDataLD1 = current_in_valid_sorted;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.8, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    elseif strcmp(cfg.str_data_day, '20221024') && strcmp(str_side, 'Left')
        str_stn_band = output_curr.vec_str_pow_band_stn{3};

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+3-1'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 4000]; ylimLD_LD0 = [0, 1500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [1.0, 3.0]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = false;

    elseif strcmp(cfg.str_data_day, '20221024') && strcmp(str_side, 'Right')
        str_stn_band = output_curr.vec_str_pow_band_stn{1};
        str_motor_band = output_curr.vec_str_pow_band_motor{1};

        % note that because this day was driven by gamma which is placed at
        % LD0, state_remove34 actually corresponds to gamma
        state_valid_sorted_remove34_corr = state_valid_sorted_remove14;
        state_valid_sorted_remove34_corr(state_valid_sorted_remove14 == 3) = 1;

        current_in_valid_sorted_remove34_corr = ...
            NaN(size(state_valid_sorted_remove34_corr));
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 0) = lowStimLevel;
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 1) = highStimLevel;

        state_valid_sorted_remove14_corr = state_valid_sorted_remove34;
        state_valid_sorted_remove14_corr(state_valid_sorted_remove34 == 1) = 3;
        current_in_valid_sorted_remove14_corr = ...
            NaN(size(state_valid_sorted_remove14_corr));
        current_in_valid_sorted_remove14_corr...
            (state_valid_sorted_remove14_corr == 0) = highStimLevel;
        current_in_valid_sorted_remove14_corr...
            (state_valid_sorted_remove14_corr == 3) = lowStimLevel;

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld1_valid_sorted;
        LDData_LD0 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld1_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34_corr;
        currentDataLD0 = current_in_valid_sorted_remove34_corr;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.6, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1  = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld0_valid_sorted;
        LDData_LD1 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld0_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14_corr;
        currentDataLD1 = current_in_valid_sorted_remove14_corr;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.6, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    elseif strcmp(cfg.str_data_day, '20221026') && strcmp(str_side, 'Left')
        str_stn_band = output_curr.vec_str_pow_band_stn{3};

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+3-1'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 4000]; ylimLD_LD0 = [0, 1500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [1.0, 3.0]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = false;

    elseif strcmp(cfg.str_data_day, '20221026') && strcmp(str_side, 'Right')
        str_stn_band = output_curr.vec_str_pow_band_stn{1};
        str_motor_band = output_curr.vec_str_pow_band_motor{1};

        % note that because this day was driven by gamma which is placed at
        % LD0, state_remove34 actually corresponds to gamma
        state_valid_sorted_remove34_corr = state_valid_sorted_remove14;
        state_valid_sorted_remove34_corr(state_valid_sorted_remove14 == 3) = 1;

        current_in_valid_sorted_remove34_corr = ...
            NaN(size(state_valid_sorted_remove34_corr));
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 0) = lowStimLevel;
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 1) = highStimLevel;

        state_valid_sorted_remove14_corr = state_valid_sorted_remove34;
        state_valid_sorted_remove14_corr(state_valid_sorted_remove34 == 1) = 3;
        current_in_valid_sorted_remove14_corr = ...
            NaN(size(state_valid_sorted_remove14_corr));
        current_in_valid_sorted_remove14_corr...
            (state_valid_sorted_remove14_corr == 0) = highStimLevel;
        current_in_valid_sorted_remove14_corr...
            (state_valid_sorted_remove14_corr == 3) = lowStimLevel;

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld1_valid_sorted;
        LDData_LD0 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld1_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34_corr;
        currentDataLD0 = current_in_valid_sorted_remove34_corr;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.6, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1  = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld0_valid_sorted;
        LDData_LD1 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld0_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14_corr;
        currentDataLD1 = current_in_valid_sorted_remove14_corr;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.6, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    elseif strcmp(cfg.str_data_day, '20221028') && strcmp(str_side, 'Left')
        str_stn_band = output_curr.vec_str_pow_band_stn{3};

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+3-1'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 4000]; ylimLD_LD0 = [0, 1500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [1.0, 3.0]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = false;

    elseif strcmp(cfg.str_data_day, '20221028') && strcmp(str_side, 'Right')
        str_stn_band = output_curr.vec_str_pow_band_stn{1};
        str_motor_band = output_curr.vec_str_pow_band_motor{1};

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.8, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.6, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    elseif strcmp(cfg.str_data_day, '20221030') && strcmp(str_side, 'Left')
        str_stn_band = output_curr.vec_str_pow_band_stn{3};

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+3-1'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 4000]; ylimLD_LD0 = [0, 1500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [1.0, 3.0]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = false;

    elseif strcmp(cfg.str_data_day, '20221030') && strcmp(str_side, 'Right')
        % combo day

        str_stn_band = output_curr.vec_str_pow_band_stn{1};
        str_motor_band = output_curr.vec_str_pow_band_motor{1};

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        % preprocess the LD combo data
        vecLDComboValidSortedOrig = vecLDComboValidSorted;
        vecLDComboValidSorted(vecLDComboValidSorted > 2^31) = ...
            vecLDComboValidSorted(vecLDComboValidSorted > 2^31) - 2^32;

        % next obtain the LD features
        timeLD_LDCombo = t_adap_abs_ld0_valid_sorted;
        LDData_LD0Input = ld0_data_valid_sorted(:, 1);
        LDData_LD1Input = ld0_data_valid_sorted(:, 2);
        LDData_LDCombo = vecLDComboValidSorted;
        LDThresh_LDCombo = ld0_thres_low;

        % finally obtain the state and current information
        timeStateLDCombo = t_adap_valid_sorted;
        stateDataLDCombo = state_valid_sorted_remove34;
        currentDataLDCombo = current_in_valid_sorted;
        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LDCombo = 'LD Combo'; strLD_LD0 = 'LD0 Input'; strLD_LD1 = 'LD1 Input';
        strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

        colorLDCombo = [0, 0.4470, 0.7410]; smoothColorLDCombo = [1, 0, 0];
        colorLD0 = [0, 0.4470, 0.7410]; colorLD1 = [0.8500, 0.3250, 0.0980];
        ylimLD_LDCombo = [-300, 2000];
        ylimPowLD0 = [0, 2000]; ylimLD_LD0 = [0, 500];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100];
        ylimState_LDCombo = [-0.5, 1.5]; ylimCurrent_LDCombo = [2.6, 3.6];
        ylimMDRange_LDCombo  = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = true;
        timePowLD1_Actual = t_pow_abs_motor_sorted;
        powDataLD1_Actual = pow_data_motor_sorted(:, 1);

        timeLD_LD1_Actual = t_adap_abs_ld1_valid_sorted;
        LDData_LD1_Actual = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        strLD_LD1_Actual = 'LD1';
        colorLD1_Actual = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.6, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

    elseif strcmp(cfg.str_data_day, '20221101') && strcmp(str_side, 'Left')
        str_stn_band = output_curr.vec_str_pow_band_stn{3};

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+3-1'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 4000]; ylimLD_LD0 = [0, 1500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [1.0, 3.0]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = false;

    elseif strcmp(cfg.str_data_day, '20221101') && strcmp(str_side, 'Right')
        str_stn_band = output_curr.vec_str_pow_band_stn{1};
        str_motor_band = output_curr.vec_str_pow_band_motor{1};

        % note that because this day was driven by gamma which is placed at
        % LD0, state_remove34 actually corresponds to gamma
        state_valid_sorted_remove34_corr = state_valid_sorted_remove14;
        state_valid_sorted_remove34_corr(state_valid_sorted_remove14 == 3) = 1;

        current_in_valid_sorted_remove34_corr = ...
            NaN(size(state_valid_sorted_remove34_corr));
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 0) = lowStimLevel;
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 1) = highStimLevel;

        state_valid_sorted_remove14_corr = state_valid_sorted_remove34;
        state_valid_sorted_remove14_corr(state_valid_sorted_remove34 == 1) = 3;
        current_in_valid_sorted_remove14_corr = ...
            NaN(size(state_valid_sorted_remove14_corr));
        current_in_valid_sorted_remove14_corr...
            (state_valid_sorted_remove14_corr == 0) = highStimLevel;
        current_in_valid_sorted_remove14_corr...
            (state_valid_sorted_remove14_corr == 3) = lowStimLevel;

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld1_valid_sorted;
        LDData_LD0 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld1_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34_corr;
        currentDataLD0 = current_in_valid_sorted_remove34_corr;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.6, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1  = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld0_valid_sorted;
        LDData_LD1 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld0_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14_corr;
        currentDataLD1 = current_in_valid_sorted;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.6, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

        % round 4 left side
    elseif any(strcmp(cfg.str_data_day, {'20221114AM', '20221114PM', ...
            '20221116AM', '20221116PM', '20221118AM', '20221118PM', ...
            '20221120', '20221121', '20221123AM', '20221123PM'})) ...
            && strcmp(str_side, 'Left')

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        if strcmp(cfg.str_data_day, '20221120')
            currentDataLD0 = current_in_valid_sorted_remove34;
        else
            currentDataLD0= current_in_valid_sorted;
        end

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        % for all sessions with STN beta, select right channel and PB name
        if ~any(strcmp(cfg.str_data_day, {'20221123PM'}))
            str_stn_band = output_curr.vec_str_pow_band_stn{3};
            str_motor_band = output_curr.vec_str_pow_band_stn{3};

            strCh_LD0 = '+3-1'; ylimPowLD0 = [0, 4000]; ylimLD_LD0 = [0, 1500];
            strCh_LD1 = '+3-1'; ylimPowLD1 = [0, 4000]; ylimLD_LD1 = [0, 1500];

            % for all sessions with cortical gamma
        else
            str_stn_band = output_curr.vec_str_pow_band_motor{1};
            str_motor_band = output_curr.vec_str_pow_band_motor{1};

            strCh_LD0 = '+9-8'; ylimPowLD0 = [0, 500]; ylimLD_LD0 = [0, 100];
            strCh_LD1 = '+9-8'; ylimPowLD1 = [0, 500]; ylimLD_LD1 = [0, 100];
        end

        strLD_LD0 = 'LD0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = true; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [1.0, 3.0]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1  = true;
        timePowLD1 = t_pow_abs_stn_valid_sorted;
        powDataLD1 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0, 0.4470, 0.7410]; smoothColorLD1 = [1, 0, 0];
        ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [1.0, 3.0]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

        % round 4 right side
    elseif any(strcmp(cfg.str_data_day, {'20221114AM', '20221114PM', ...
            '20221116AM', '20221116PM', '20221118AM', '20221118PM', ...
            '20221120', '20221121', '20221123AM'}))...
            && strcmp(str_side, 'Right')
        % combo day
        str_stn_band = output_curr.vec_str_pow_band_stn{1};
        str_motor_band = output_curr.vec_str_pow_band_motor{2};

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 2);

        % preprocess the LD combo data
        vecLDComboValidSortedOrig = vecLDComboValidSorted;
        vecLDComboValidSorted(vecLDComboValidSorted > 2^31) = ...
            vecLDComboValidSorted(vecLDComboValidSorted > 2^31) - 2^32;

        % next obtain the LD features
        timeLD_LDCombo = t_adap_abs_ld0_valid_sorted;
        LDData_LD0Input = ld0_data_valid_sorted(:, 1);
        LDData_LD1Input = ld0_data_valid_sorted(:, 2);
        LDData_LDCombo = vecLDComboValidSorted;
        LDThresh_LDCombo = ld0_thres_low;

        % finally obtain the state and current information
        timeStateLDCombo = t_adap_valid_sorted;
        stateDataLDCombo = state_valid_sorted_remove34;
        if strcmp(cfg.str_data_day, '20221120')
            currentDataLDCombo = current_in_valid_sorted_remove34;
        else
            currentDataLDCombo = current_in_valid_sorted;
        end
        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LDCombo = 'LD Combo'; strLD_LD0 = 'LD0 Input'; strLD_LD1 = 'LD1 Input';
        strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = true; boolStreamed_LD0 = true;

        strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

        colorLDCombo = [0, 0.4470, 0.7410]; smoothColorLDCombo = [1, 0, 0];
        colorLD0 = [0, 0.4470, 0.7410]; colorLD1 = [0.8500, 0.3250, 0.0980];
        ylimLD_LDCombo = [-300, 2000];
        ylimPowLD0 = [0, 2000]; ylimLD_LD0 = [0, 500];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100];
        ylimState_LDCombo = [-0.5, 1.5]; ylimCurrent_LDCombo = [2.6, 3.6];
        ylimMDRange_LDCombo  = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = true;
        timePowLD1_Actual = t_pow_abs_motor_sorted;
        powDataLD1_Actual = pow_data_motor_sorted(:, 1);

        timeLD_LD1_Actual = t_adap_abs_ld1_valid_sorted;
        LDData_LD1_Actual = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        strLD_LD1_Actual = 'LD1';
        colorLD1_Actual = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.6, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        % round 4 right side - gamma day
    elseif any(strcmp(cfg.str_data_day, {'20221123PM'}))...
            && strcmp(str_side, 'Right')
        % gamma day
        str_stn_band = output_curr.vec_str_pow_band_stn{1};
        str_motor_band = output_curr.vec_str_pow_band_motor{2};

        % note that because this day was driven by gamma which is placed at
        % LD0, state_remove34 actually corresponds to gamma
        state_valid_sorted_remove34_corr = state_valid_sorted_remove14;
        state_valid_sorted_remove34_corr(state_valid_sorted_remove14 == 3) = 1;

        current_in_valid_sorted_remove34_corr = ...
            NaN(size(state_valid_sorted_remove34_corr));
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 0) = lowStimLevel;
        current_in_valid_sorted_remove34_corr...
            (state_valid_sorted_remove34_corr == 1) = highStimLevel;

        state_valid_sorted_remove14_corr = state_valid_sorted_remove34;
        state_valid_sorted_remove14_corr(state_valid_sorted_remove34 == 1) = 3;
        current_in_valid_sorted_remove14_corr = ...
            NaN(size(state_valid_sorted_remove14_corr));
        current_in_valid_sorted_remove14_corr...
            (state_valid_sorted_remove14_corr == 0) = highStimLevel;
        current_in_valid_sorted_remove14_corr...
            (state_valid_sorted_remove14_corr == 3) = lowStimLevel;

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 2);

        % preprocess the LD combo data
        vecLDComboValidSortedOrig = vecLDComboValidSorted;
        vecLDComboValidSorted(vecLDComboValidSorted > 2^31) = ...
            vecLDComboValidSorted(vecLDComboValidSorted > 2^31) - 2^32;

        % next obtain the LD features
        timeLD_LDCombo = t_adap_abs_ld1_valid_sorted;
        LDData_LD0Input = ld1_data_valid_sorted(:, 1);
        LDData_LD1Input = ld1_data_valid_sorted(:, 2);
        LDData_LDCombo = vecLDComboValidSorted;
        LDThresh_LDCombo = ld1_thres_low;

        % finally obtain the state and current information
        timeStateLDCombo = t_adap_valid_sorted;
        stateDataLDCombo = state_valid_sorted_remove34_corr;
        currentDataLDCombo = current_in_valid_sorted_remove34_corr;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LDCombo = 'LD Combo'; strLD_LD0 = 'LD0 Input'; strLD_LD1 = 'LD1 Input';
        strCh_LD0 = '+2-0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = false; boolStreamed_LD0 = true;

        strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = true; boolStreamed_LD1 = true;

        colorLDCombo = [0, 0.4470, 0.7410]; smoothColorLDCombo = [1, 0, 0];
        colorLD0 = [0, 0.4470, 0.7410]; colorLD1 = [0.8500, 0.3250, 0.0980];
        ylimLD_LDCombo = [-300, 2000];
        ylimPowLD0 = [0, 2000]; ylimLD_LD0 = [0, 500];
        ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100];
        ylimState_LDCombo = [-0.5, 1.5]; ylimCurrent_LDCombo = [2.6, 3.6];
        ylimMDRange_LDCombo  = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1 = true;
        timePowLD1_Actual = t_pow_abs_motor_sorted;
        powDataLD1_Actual = pow_data_motor_sorted(:, 1);

        timeLD_LD1_Actual = t_adap_abs_ld0_valid_sorted;
        LDData_LD1_Actual = ld0_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld0_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14_corr;
        currentDataLD1 = current_in_valid_sorted;

        strLD_LD1_Actual = 'LD1';
        colorLD1_Actual = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.6, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        % round 5 left side
    elseif any(strcmp(cfg.str_data_day, {'20221130AM', '20221130PM', ...
            '20221202AM', '20221202PM', '20221204', '20221208', ...
            '20221209'})) ...
            && strcmp(str_side, 'Left')

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0= current_in_valid_sorted;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        % for all sessions with cortical gamma

        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500];
        ylimPowLD1 = [0, 1500]; ylimLD_LD1 = [0, 500];

        strLD_LD0 = LD0_adaptiveMetaData.strLD; 
        strCh_LD0 = LD0_adaptiveMetaData.strCh{1}; 
        strPB_LD0 = LD0_adaptiveMetaData.strPB{1};
        boolActiveLD_LD0 = LD0_adaptiveMetaData.boolActive; 
        boolStreamed_LD0 = LD0_adaptiveMetaData.boolEnabled;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [1.0, 3.0]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1  = true;
        timePowLD1 = t_pow_abs_stn_valid_sorted;
        powDataLD1 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0, 0.4470, 0.7410]; smoothColorLD1 = [1, 0, 0];
        ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [1.0, 3.0]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = LD1_adaptiveMetaData.strLD; 
        strCh_LD1 = LD1_adaptiveMetaData.strCh{1}; 
        strPB_LD1 = LD1_adaptiveMetaData.strPB{1};
        boolActiveLD_LD1 = LD1_adaptiveMetaData.boolActive; 
        boolStreamed_LD1 = LD1_adaptiveMetaData.boolEnabled;

        % round 5 right side - gamma day
    elseif any(strcmp(cfg.str_data_day, {'20221130AM', '20221130PM', ...
            '20221202AM', '20221202PM', '20221204', '20221208', ...
            '20221209'}))...
            && strcmp(str_side, 'Right')

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = LD0_adaptiveMetaData.strLD; 
        strCh_LD0 = LD0_adaptiveMetaData.strCh{1}; 
        strPB_LD0 = LD0_adaptiveMetaData.strPB{1};
        boolActiveLD_LD0 = LD0_adaptiveMetaData.boolActive; 
        boolStreamed_LD0 = LD0_adaptiveMetaData.boolEnabled;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.6, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1  = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 1500]; ylimLD_LD1 = [0, 500]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [2.6, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = LD1_adaptiveMetaData.strLD; 
        strCh_LD1 = LD1_adaptiveMetaData.strCh{1}; 
        strPB_LD1 = LD1_adaptiveMetaData.strPB{1};
        boolActiveLD_LD1 = LD1_adaptiveMetaData.boolActive; 
        boolStreamed_LD1 = LD1_adaptiveMetaData.boolEnabled;

        % round 5 left side with dual threshold
    elseif any(strcmp(cfg.str_data_day, {'20221205'})) ...
            && strcmp(str_side, 'Left')

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0= current_in_valid_sorted;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        % for all sessions with cortical gamma
        str_stn_band = output_curr.vec_str_pow_band_motor{1};
        str_motor_band = output_curr.vec_str_pow_band_motor{1};

        strCh_LD0 = '+9-8'; ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500];
        strCh_LD1 = '+9-8'; ylimPowLD1 = [0, 1500]; ylimLD_LD1 = [0, 500];

        strLD_LD0 = 'LD0'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = true; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [1.0, 3.0]; ylimMDRange_LD0 = [-0.5, 6.5];

        % LD1 parameters
        boolEnabled_LD1  = true;
        timePowLD1 = t_pow_abs_stn_valid_sorted;
        powDataLD1 = pow_data_stn_valid_sorted(:, 3);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0, 0.4470, 0.7410]; smoothColorLD1 = [1, 0, 0];
        ylimState_LD1 = [-0.5, 6.5];
        ylimCurrent_LD1 = [1.0, 3.0]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

        % round 5 right side - gamma day with dual threshold
    elseif any(strcmp(cfg.str_data_day, {'20221205'}))...
            && strcmp(str_side, 'Right')
        str_stn_band = output_curr.vec_str_pow_band_motor{1};
        str_motor_band = output_curr.vec_str_pow_band_motor{1};

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = true;
        timePowLD0 = t_pow_abs_stn_valid_sorted;
        powDataLD0 = pow_data_stn_valid_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);
        LDThresh_LD0 = ld0_thres_low;

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted;

        timeAWDysk = NaN; AWDyskData = NaN;
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        strLD_LD0 = 'LD0'; strCh_LD0 = '+9-8'; strPB_LD0 = str_stn_band;
        boolActiveLD_LD0 = true; boolStreamed_LD0 = true;

        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 1500]; ylimLD_LD0 = [0, 500]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [2.6, 3.6]; ylimMDRange_LD0 = [-0.5, 3.5];

        % LD1 parameters
        boolEnabled_LD1  = true;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 1);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);
        LDThresh_LD1 = ld1_thres_low;

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 1500]; ylimLD_LD1 = [0, 500];
        ylimState_LD1 = [-0.5, 6.5];
        ylimCurrent_LD1 = [2.6, 3.6]; ylimMDRange_LD1 = [-0.5, 3.5];

        strLD_LD1 = 'LD1'; strCh_LD1 = '+9-8'; strPB_LD1 = str_motor_band;
        boolActiveLD_LD1 = false; boolStreamed_LD1 = true;

    else
        error("need to specify a date")
    end

    %% custom flags for RCS14
elseif strcmp(cfg.str_sub, 'RCS14')
    if any(strcmp(cfg.str_data_day, {'20221024', '20221026', '20221028', ...
            '20221031', '20221101', '20230117', '20230118', '20230119', ...
            '20230321'}))
        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = LD0_adaptiveMetaData.boolEnabled;
        timePowLD0 = t_pow_abs_motor_sorted;
        powDataLD0 = pow_data_motor_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        if ~any(strcmp(cfg.str_data_day, cfg.str_no_aw_data_day))
            timeAWDysk = t_accel_full; AWDyskData = accel_full;
        else
            timeAWDysk = NaN; AWDyskData = NaN;
        end
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;
        
        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 10000]; ylimLD_LD0 = [0, 60]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [3.4, 4.2]; ylimMDRange_LD0 = [-0.5, 5.5];

        % LD1 parameters
        boolEnabled_LD1 = LD1_adaptiveMetaData.boolEnabled;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 2);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 7000]; ylimLD_LD1 = [0, 5e4]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [3.4, 4.2]; ylimMDRange_LD1 = [-0.5, 5.5];
    

    elseif any(strcmp(cfg.str_data_day, {'20230426', '20230428'}))
        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = LD0_adaptiveMetaData.boolEnabled;
        timePowLD0 = t_pow_abs_motor_sorted;
        powDataLD0 = pow_data_motor_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        if ~any(strcmp(cfg.str_data_day, cfg.str_no_aw_data_day))
            timeAWDysk = t_accel_full; AWDyskData = accel_full;
        else
            timeAWDysk = NaN; AWDyskData = NaN;
        end
        timePKG = pkgWatchTableRedacted.Date_Time;
        PKGBradyData = pkgWatchTableRedacted.BK;
        PKGDyskData = pkgWatchTableRedacted.DK;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;
        
        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 10000]; ylimLD_LD0 = [0, 60]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [3.4, 4.2]; ylimMDRange_LD0 = [-0.5, 5.5];

        % LD1 parameters
        boolEnabled_LD1 = LD1_adaptiveMetaData.boolEnabled;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 2);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 7000]; ylimLD_LD1 = [0, 5e4]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [3.4, 4.2]; ylimMDRange_LD1 = [-0.5, 5.5];

    else
        error("need to specify a date")
    end

        %% custom flags for RCS17
elseif strcmp(cfg.str_sub, 'RCS17')
    if any(strcmp(cfg.str_data_day, {'20230522AM', '20230522PM'}))
        % getting the raw data as well
        timeRaw = t_raw_data_full;
        rawData = raw_data_full;

        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = LD0_adaptiveMetaData.boolEnabled;
        timePowLD0 = t_pow_abs_motor_sorted;
        powDataLD0 = pow_data_motor_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        if ~any(strcmp(cfg.str_data_day, cfg.str_no_aw_data_day))
            timeAWDysk = t_accel_full; AWDyskData = accel_full;
        else10
            timeAWDysk = NaN; AWDyskData = NaN;
        end
        timePKG = NaN;
        PKGBradyData = NaN;
        PKGDyskData = NaN;
        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;
        
        colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
        ylimPowLD0 = [0, 10000]; ylimLD_LD0 = [0, 60]; ylimState_LD0 = [-0.5, 1.5];
        ylimCurrent_LD0 = [3.4, 4.2]; ylimMDRange_LD0 = [-0.5, 5.5];

        % LD1 parameters
        boolEnabled_LD1 = LD1_adaptiveMetaData.boolEnabled;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 2);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;

        colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
        ylimPowLD1 = [0, 7000]; ylimLD_LD1 = [0, 5e4]; ylimState_LD1 = [-0.5, 3.5];
        ylimCurrent_LD1 = [3.4, 4.2]; ylimMDRange_LD1 = [-0.5, 5.5];

        print('debug')

    elseif any(strcmp(cfg.str_data_day, {'20230524', '20230531', '20230605', ...
            '20230726', '20230729'}))
        % define the various variables for plotting
        % LD0 parameters
        timeResCxtWin = 2;
        boolEnabled_LD0 = LD0_adaptiveMetaData.boolEnabled;
        timePowLD0 = t_pow_abs_motor_sorted;
        powDataLD0 = pow_data_motor_sorted(:, 1);

        timeLD_LD0 = t_adap_abs_ld0_valid_sorted;
        LDData_LD0 = ld0_data_valid_sorted(:, 1);

        timeStateLD0 = t_adap_valid_sorted;
        stateDataLD0 = state_valid_sorted_remove34;
        currentDataLD0 = current_in_valid_sorted_remove34;

        if ~any(strcmp(cfg.str_data_day, cfg.str_no_aw_data_day))
            timeAWDysk = t_accel_full; AWDyskData = accel_full;
        else
            timeAWDysk = NaN; AWDyskData = NaN;
        end
            
        if ~any(strcmp(cfg.str_data_day, cfg.str_no_pkg_data_day))
            timePKG = pkgWatchTable.Date_Time;
            PKGBradyData = pkgWatchTable.BK;
            PKGDyskData = pkgWatchTable.DK;
        else
            timePKG = NaN;
            PKGBradyData = NaN;
            PKGDyskData = NaN;
        end


%         % debug script
%         figure; subplot(311); plot(timeLD_LD0, LDData_LD0); title("LD0")
%         subplot(312); plot(timeStateLD0, stateDataLD0); title('States'); ylim([-0.5, 1.5]);
%         subplot(313); plot(timeStateLD0, currentDataLD0); title('Current'); ylim([1.1, 3.9])
% 

        motorDiaryInterp = motorDiaryInterpRedacted;
        motorDiary = motorDiaryRedacted;

        % sanity check
        if numel(cfg.vec_str_side) == 2
            error('Only provide one side')
        end

        if any(strcmp(cfg.vec_str_side, {'Left'}))
            colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
            ylimPowLD0 = [0, 10000]; ylimLD_LD0 = [0, 15000]; ylimState_LD0 = [-0.5, 1.5];
            ylimCurrent_LD0 = [1.4, 3.2]; ylimMDRange_LD0 = [-0.5, 4.5];
        elseif any(strcmp(cfg.vec_str_side, {'Right'}))
            colorLD0 = [0, 0.4470, 0.7410]; smoothColorLD0 = [1, 0, 0];
            ylimPowLD0 = [0, 10000]; ylimLD_LD0 = [0, 1200]; ylimState_LD0 = [-0.5, 1.5];
            ylimCurrent_LD0 = [1.5, 2.7]; ylimMDRange_LD0 = [-0.5, 4.5];
        end

        % LD1 parameters
        boolEnabled_LD1 = LD1_adaptiveMetaData.boolEnabled;
        timePowLD1 = t_pow_abs_motor_sorted;
        powDataLD1 = pow_data_motor_sorted(:, 2);

        timeLD_LD1 = t_adap_abs_ld1_valid_sorted;
        LDData_LD1 = ld1_data_valid_sorted(:, 1);

        timeStateLD1 = t_adap_valid_sorted;
        stateDataLD1 = state_valid_sorted_remove14;
        currentDataLD1 = current_in_valid_sorted_remove14;
        
        if any(strcmp(cfg.vec_str_side, {'Left'}))
            colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
            ylimPowLD1 = [0, 7000]; ylimLD_LD1 = [0, 15000]; ylimState_LD1 = [-0.5, 3.5];
            ylimCurrent_LD1 = [2, 3.2]; ylimMDRange_LD1 = [-0.5, 4.5];
        elseif any(strcmp(cfg.vec_str_side, {'Right'}))
            colorLD1 = [0.3010 0.7450 0.9330]; smoothColorLD1 = [1, 0, 0];
            ylimPowLD1 = [0, 7000]; ylimLD_LD1 = [0, 4000]; ylimState_LD1 = [-0.5, 3.5];
            ylimCurrent_LD1 = [1.5, 2.7]; ylimMDRange_LD1 = [-0.5, 4.5];
        end
    end
end
%% compute some output statistics

% compute mean and standard deviation of current
summaryOutput.avgCurrentIn = mean(current_in_valid_sorted);
summaryOutput.stdCurrentIn = std(current_in_valid_sorted);

% compute percentage of high and low stim
summaryOutput.perCurrentHigh = sum(current_in_valid_sorted == highStimLevel) / ...
    size(current_in_valid_sorted, 1);
summaryOutput.perCurrentLow = sum(current_in_valid_sorted == lowStimLevel) / ...
    size(current_in_valid_sorted, 1);

%% Plotting
% plot normally for non-combo threshold cases
if ~cfg.thresCombo

    %% Plot the different subplots for STN
    if boolEnabled_LD0
        % plot the fluctuation plots for LD0 for full time duration
        [figTD_LD0, fFigure_Fluc_LD0, summaryOutput] = ...
            plotTDFluc(timePowLD0, powDataLD0, timeLD_LD0, LDData_LD0, ...
            timeStateLD0, stateDataLD0, currentDataLD0, ...
            timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
            motorDiaryInterp, motorDiary, ...
            medTime, vecDyskOnsetFull, vecDyskOffsetFull, ...
            LD0_adaptiveMetaData, str_side, cfg, ...
            'color', colorLD0, ...
            'smoothColor', smoothColorLD0, 'ylimPow', ylimPowLD0 ,...
            'ylimLD', ylimLD_LD0, 'ylimState', ylimState_LD0, ...
            'ylimCurrent', ylimCurrent_LD0, 'ylimMDRange', ylimMDRange_LD0, ...
            'boolReturnOutput', false, 'outputStruct', summaryOutput);
    
        if boolSaveAsFig
            savefig(figTD_LD0, fullfile(pFigure, fFigure_Fluc_LD0));
            close(figTD_LD0)
        else
            saveas(figTD_LD0, fullfile(pFigure, sprintf('%s.png', fFigure_Fluc_LD0)));
            close(figTD_LD0);
        end
        
        % now plot the changes over time in finer windows
        plotTDFlucOverTimeCxtWin(timeResCxtWin, ...
            timePowLD0, powDataLD0, timeLD_LD0, LDData_LD0, ...
            timeStateLD0, stateDataLD0, currentDataLD0, ...
            timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
            motorDiaryInterp, motorDiary, ...
            medTime, vecDyskOnsetFull, vecDyskOffsetFull, ...
            LD0_adaptiveMetaData, str_side, ...
            pFigure, cfg, ...
            'color', colorLD0, ...
            'smoothColor', smoothColorLD0, 'ylimPow', ylimPowLD0 ,...
            'ylimLD', ylimLD_LD0, 'ylimState', ylimState_LD0, ...
            'ylimCurrent', ylimCurrent_LD0, 'ylimMDRange', ylimMDRange_LD0, ...
            'boolSaveAsFig', boolSaveAsFig);
        
        % Now also plot the stacked bar plots for LD0
        
        % now estimate the total proportion of stim in each state
        [figBox_LD0, fFigure_Box_LD0, summaryOutput] = ...
            plotBoxPlot(timeStateLD0, stateDataLD0, motorDiaryRedacted, ...
            vecDyskOnsetFull, vecDyskOffsetFull, LD0_adaptiveMetaData, cfg, ...
            'boolReturnOutput', true, 'outputStruct', summaryOutput);
        
        saveas(figBox_LD0, fullfile(pFigure, sprintf('%s.png', fFigure_Box_LD0)));
        close(figBox_LD0);
    
        % hardcode the output directory for summary stats
        pSummaryStats = fullfile('/home/jyao/local/data/starrlab/proc_data', ...
            cfg.str_sub, cfg.str_paradigm, cfg.str_aDBS_paradigm, ...
            'all_days');
        save(fullfile(pSummaryStats, sprintf('summary_%s_%s.mat', ...
            cfg.str_data_day, str_side(1))), "summaryOutput");

%         plotWearableComp(timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
%             motorDiaryInterp, motorDiary, ...
%             medTime, vecDyskOnsetFull, vecDyskOffsetFull)
    end
    
    %% Now also plot the LD1's result
    if boolEnabled_LD1
        % plot the fluctuation plots for LD1 for full time duration
        [figTD_LD1, fFigure_Fluc_LD1, ~] = ...
            plotTDFluc(timePowLD1, powDataLD1, timeLD_LD1, LDData_LD1, ...
            timeStateLD1, stateDataLD1, currentDataLD1, ...
            timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
            motorDiaryInterp, motorDiary, ...
            medTime, vecDyskOnsetFull, vecDyskOffsetFull, ...
            LD1_adaptiveMetaData, str_side, cfg, ...
            'color', colorLD1, ...
            'smoothColor', smoothColorLD1, 'ylimPow', ylimPowLD1 ,...
            'ylimLD', ylimLD_LD1, 'ylimState', ylimState_LD1, ...
            'ylimCurrent', ylimCurrent_LD1, 'ylimMDRange', ylimMDRange_LD1);
        
        if boolSaveAsFig
            savefig(figTD_LD1, fullfile(pFigure, fFigure_Fluc_LD1));
            close(figTD_LD1);
        else
            saveas(figTD_LD1, fullfile(pFigure, sprintf('%s.png', fFigure_Fluc_LD1)));
            close(figTD_LD1);
        end
        
        % now plot the changes over time in finer windows
        plotTDFlucOverTimeCxtWin(timeResCxtWin, ...
            timePowLD1, powDataLD1, timeLD_LD1, LDData_LD1, ...
            timeStateLD1, stateDataLD1, currentDataLD1, ...
            timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
            motorDiaryInterp, motorDiary, ...
            medTime, vecDyskOnsetFull, vecDyskOffsetFull, ...
            LD1_adaptiveMetaData, str_side, ...
            pFigure, cfg, ...
            'color', colorLD1, ...
            'smoothColor', smoothColorLD1, 'ylimPow', ylimPowLD1 ,...
            'ylimLD', ylimLD_LD1, 'ylimState', ylimState_LD1, ...
            'ylimCurrent', ylimCurrent_LD1, 'ylimMDRange', ylimMDRange_LD1, ...
            'boolSaveAsFig', boolSaveAsFig);
        
        % now plot the proportion of LD1-based states
        
        % now estimate the total proportion of stim in each state
        [figBox_LD1, fFigure_Box_LD1, ~] = ...
            plotBoxPlot(timeStateLD1, stateDataLD1, motorDiaryRedacted, ...
            vecDyskOnsetFull, vecDyskOffsetFull, LD1_adaptiveMetaData, cfg);
        
        saveas(figBox_LD1, fullfile(pFigure, sprintf('%s.png', fFigure_Box_LD1)));
        close(figBox_LD1);
    end

else
    %% if the case of combo
    % plot the fluctuation plots for LD1 for full time duration
    [figTD_LDCombo, fFigure_Fluc_LDCombo, summaryOutput] = ...
        plotTDFlucCombo(timePowLD0, powDataLD0, timePowLD1, powDataLD1, ...
        timeLD_LDCombo, LDData_LDCombo, LDData_LD0Input, LDData_LD1Input, LDThresh_LDCombo, ...
        timeStateLDCombo, stateDataLDCombo, currentDataLDCombo, ...
        timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
        motorDiaryInterp, motorDiary, ...
        medTime, vecDyskOnsetFull, vecDyskOffsetFull, strLD_LDCombo,...
        strLD_LD0, strCh_LD0, strPB_LD0, ...
        strLD_LD1, strCh_LD1, strPB_LD1, cfg, ...
        'boolStreamed_LD0', boolStreamed_LD0, 'boolStreamed_LD1', boolStreamed_LD1, ...
        'colorLDCombo', colorLDCombo, 'colorLD0', colorLD0, ...
        'colorLD1', colorLD1, 'smoothColor', smoothColorLDCombo, ...
        'ylimPowLD0', ylimPowLD0 , 'ylimPowLD1', ylimPowLD1, ...
        'ylimLD_LDCombo', ylimLD_LDCombo, 'ylimLD_LD0', ylimLD_LD0, ...
        'ylimLD_LD1', ylimLD_LD1, 'ylimState', ylimState_LDCombo, ...
        'ylimCurrent', ylimCurrent_LDCombo, 'ylimMDRange', ylimMDRange_LDCombo, ...
        'boolReturnOutput', true, 'outputStruct', summaryOutput);
    
    if boolSaveAsFig
        savefig(figTD_LDCombo, fullfile(pFigure, fFigure_Fluc_LDCombo));
        close(figTD_LDCombo)
    else
        saveas(figTD_LDCombo, fullfile(pFigure, sprintf('%s.png', fFigure_Fluc_LDCombo)));
        close(figTD_LDCombo)
    end

    % now estimate the total proportion of stim in each state
    [figBox_LDCombo, fFigure_Box_LDCombo, summaryOutput] = ...
        plotBoxPlot(timeStateLDCombo, stateDataLDCombo, motorDiaryRedacted, ...
        vecDyskOnsetFull, vecDyskOffsetFull, LD0_adaptiveMetaData, cfg, ...
        'boolReturnOutput', true, 'outputStruct', summaryOutput);
    
    saveas(figBox_LDCombo, fullfile(pFigure, sprintf('%s.png', fFigure_Box_LDCombo)));
    close(figBox_LDCombo);

    pSummaryStats = fullfile('/home/jyao/local/data/starrlab/proc_data', ...
        cfg.str_sub, cfg.str_paradigm, cfg.str_aDBS_paradigm, ...
        'all_days');
    save(fullfile(pSummaryStats, sprintf('summary_%s_%s.mat', ...
        cfg.str_data_day, str_side(1))), 'summaryOutput');

    % for some dates also plot the LD1
    if any(strcmp(cfg.str_combo_w_LD1, cfg.str_data_day))
            
        if boolEnabled_LD1

            if any(strcmp(cfg.str_data_day, {'20221114AM', '20221114PM', ...
                    '20221116AM', '20221116PM', '20221118AM', '20221118PM', ...
                    '20221120', '20221121'}))
                ylimPowLD1 = [0, 1500]; ylimLD_LD1 = [0, 500];

            elseif any(strcmp(cfg.str_data_day, {'20221123AM', '20221123PM', ...
                    '20221130AM'}))
                ylimPowLD1 = [0, 300]; ylimLD_LD1 = [0, 100];
            end

            % plot the fluctuation plots for LD1 for full time duration
            [figTD_LD1, fFigure_Fluc_LD1, ~] = ...
                plotTDFluc(timePowLD1_Actual, powDataLD1_Actual, ...
                timeLD_LD1_Actual, LDData_LD1_Actual, ...
                timeStateLD1, stateDataLD1, currentDataLD1, ...
                timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
                motorDiaryInterp, motorDiary, ...
                medTime, vecDyskOnsetFull, vecDyskOffsetFull, ...
                LD1_adaptiveMetaData, cfg, ...
                'color', colorLD1_Actual, ...
                'smoothColor', smoothColorLD1, 'ylimPow', ylimPowLD1 ,...
                'ylimLD', ylimLD_LD1, 'ylimState', ylimState_LD1, ...
                'ylimCurrent', ylimCurrent_LD1, 'ylimMDRange', ylimMDRange_LD1);
            
            if boolSaveAsFig
                savefig(figTD_LD1, fullfile(pFigure, fFigure_Fluc_LD1));
                close(figTD_LD1);
            else
                saveas(figTD_LD1, fullfile(pFigure, sprintf('%s.png', fFigure_Fluc_LD1)));
                close(figTD_LD1);
            end
            
            % now plot the changes over time in finer windows
            plotTDFlucOverTimeCxtWin(timeResCxtWin, ...
                timePowLD1_Actual, powDataLD1_Actual, ...
                timeLD_LD1_Actual, LDData_LD1_Actual, ...
                timeStateLD1, stateDataLD1, currentDataLD1, ...
                timeAWDysk, AWDyskData, timePKG, PKGBradyData, PKGDyskData, ...
                motorDiaryInterp, motorDiary, ...
                medTime, vecDyskOnsetFull, vecDyskOffsetFull, ...
                LD1_adaptiveMetaData, str_side, ...
                pFigure, cfg, ...
                'color', colorLD1_Actual, ...
                'smoothColor', smoothColorLD1, 'ylimPow', ylimPowLD1 ,...
                'ylimLD', ylimLD_LD1, 'ylimState', ylimState_LD1, ...
                'ylimCurrent', ylimCurrent_LD1, 'ylimMDRange', ylimMDRange_LD1, ...
                'boolSaveAsFig', boolSaveAsFig);
            
            % now plot the proportion of LD1-based states
            
            % now estimate the total proportion of stim in each state
            [figBox_LD1, fFigure_Box_LD1, ~] = ...
                plotBoxPlot(timeStateLD1, stateDataLD1, motorDiaryRedacted, ...
                vecDyskOnsetFull, vecDyskOffsetFull, LD1_adaptiveMetaData, cfg);
            
            saveas(figBox_LD1, fullfile(pFigure, sprintf('%s.png', fFigure_Box_LD1)));
            close(figBox_LD1);
        end

    end

end

end