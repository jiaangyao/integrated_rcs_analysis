
%% TODO: note this is deprecated
%% Envrionment setup

clearvars -except nothing;
clc;
close all;

set_env;
addpath(fullfile('..', 'analyze_rcs_data', 'code'));
addpath(fullfile('..', 'auxf', 'external'));
addpath(fullfile('..', 'auxf', 'internal'));
addpath(fullfile('preproc'))
addpath(fullfile('utils'))
addpath(fullfile('plotting'))


%% Setting the aDBS path
p_data_in = getenv('D_DATA_IN');
p_data_out = getenv('D_DATA_OUT');
p_figure = getenv('D_FIGURE');

% setting subject, paradigm and index of paradigm
% also specify the date of recording
% pick the side intended for analysis
cfg.str_sub = 'RCS08';
cfg.str_paradigm = getenv('STR_STEP6');
cfg.idx_paradigm = split(cfg.str_paradigm, '_');
cfg.idx_paradigm = cfg.idx_paradigm{1};
cfg.idx_paradigm = str2num(cfg.idx_paradigm(end));
cfg.str_data_day = '06072022';
cfg.vec_str_side = {'Left'};
cfg.overwrite = false;
cfg.figure_overwrite = true;

if strcmp(cfg.str_data_day, '3day_sprint') || contains(cfg.str_data_day, '2020')
    cfg.bool_old_recording = true;
else
    cfg.bool_old_recording = false;
end

% these variables are read off from the env file
cfg.str_data_type = getenv('STR_DATA_TYPE');
cfg.str_data_folder = getenv('STR_DATA_FOLDER');

% these variables relate to settings for each specific step
% step 3 specific parameters
if cfg.idx_paradigm == 3
    cfg.analyze_step3 = true;
else
    cfg.analyze_step3 = false;
end

% step 4 specific parameters
if cfg.idx_paradigm == 4
    cfg.analyze_step4 = true;
else
    cfg.analyze_step4 = false;
end
cfg.vec_str_low_high_stim = {'low'};
cfg.bool_in_clinic = false;

% step 5 specific parameters
if cfg.idx_paradigm == 5
    cfg.analyze_step5 = true;
else
    cfg.analyze_step5 = false;
end

% step 6 specific parameters
if cfg.idx_paradigm == 6
    cfg.analyze_step6 = true;
else
    cfg.analyze_step6 = false;
end


% cfg.DBS_paradigm = 'SCBS';
% cfg.DBS_paradigm = 'aDBS';
cfg.DBS_paradigm = 'Normal';


cfg = getDeviceName(cfg);


%% Setting analysis related hyperparameters


cfg.len_epoch_prune = 2;                                                    % length of epoch in seconds used for artifact removal
if strcmp(cfg.str_sub, 'RCS08')
    cfg.amp_thres = 500;                                                    % amplitude threshold of STN artifacts
elseif strcmp(cfg.str_sub, 'RCS02')
    cfg.amp_thres = 100;
    cfg.pac_amp_thres = 500;
end

cfg.vec_epoch_pattern = {[0, 1, 0], [0, 0, 1, 1, 0]};                       % pattern of continuous epochs to be dropped
cfg.bool_apply_lpf_stn = true;                                              % flag for whether or not to use HPF for STN data
cfg.lpf_edge_stn = 0.5;                                                     % freq in Hz for LPF for STN data
cfg.bool_apply_notch_stn = false;                                           % flag for whether or not to use notch at powerline for STN data


cfg.bool_apply_lpf_motor = true;                                            % flag for whether or not to use HPF for cortical data
cfg.lpf_edge_motor = 0.5;                                                   % freq in Hz for HPF for cortical data
cfg.bool_apply_notch_motor = false;                                         % flag for whether or not to use notch at powerline for cortical data

cfg.bool_reref = false;                                                     % flag for whether or not to use re-referencing on cortical and STN data
cfg.str_reref_mode = 'CAR';                                                 % re-referencing method to be used, either CAR or bipolar


%% Perform preprocessing

for idx_side = 1:length(cfg.vec_str_side)
    vec_output = {};

    % obtain the directory name corresponding to current side
    str_side = cfg.vec_str_side{idx_side};
    if strcmp(str_side, 'Left')
        str_sub_side_full = sprintf('%s%s', cfg.str_sub, 'L');
        str_device_curr = cfg.str_device_L;
    elseif strcmp(str_side, 'Right')
        str_sub_side_full = sprintf('%s%s', cfg.str_sub, 'R');
        str_device_curr = cfg.str_device_R;
    else
        error('invalid side');
    end

    % now obtain the full input path and get ready for glob
    % first set the handle and then adjust path based on which step it is
    p_data_in_handle = fullfile(p_data_in, cfg.str_sub, cfg.str_paradigm, ...
        cfg.str_data_day);
    if cfg.analyze_step3
        error("Not implemented yet")
    elseif cfg.analyze_step4
        error("Not implemented yet")
    elseif cfg.analyze_step5
        p_data_in_full_curr = fullfile(p_data_in_handle, cfg.str_data_type, ...
            cfg.DBS_paradigm, cfg.str_data_folder, str_sub_side_full);
        
        % set the output directory also
        p_data_out_full_curr = fullfile(p_data_out, cfg.str_sub, ...
            cfg.str_paradigm, cfg.DBS_paradigm, cfg.str_data_day, ...
            str_sub_side_full);
    elseif cfg.analyze_step6
        p_data_in_full_curr = fullfile(p_data_in_handle, cfg.str_data_type, ...
            cfg.DBS_paradigm, cfg.str_data_folder, str_sub_side_full);

        % set the output directory also
        p_data_out_full_curr = fullfile(p_data_out, cfg.str_sub, ...
            cfg.str_paradigm, cfg.DBS_paradigm, cfg.str_data_day, ...
            str_sub_side_full);
    else
        error("Need to perform one analysis")
    end

    % create the output path if DNE
    if exist(p_data_out_full_curr, 'dir') ~= 7
        mkdir(p_data_out_full_curr);
    end
    
    % globbing the full list of sessions
    vec_str_session_curr = dir(fullfile(p_data_in_full_curr, ['*', 'ession*']));

    % now loop through all available sessions
    fprintf("Processing time series data\n")
    for idx_session = 1:numel(vec_str_session_curr)
        str_session = vec_str_session_curr(idx_session).name;
        
        % check if current file has already been processed
        p_data_out_raw_curr = fullfile(p_data_out_full_curr, 'raw_struct');
        if exist(p_data_out_raw_curr, 'dir') ~= 7
            mkdir(p_data_out_raw_curr);
        end
        f_raw_struct_curr = sprintf('raw_struct_%s.mat', str_session);

        if exist(fullfile(p_data_out_raw_curr, f_raw_struct_curr), 'file') == 0 || ...
                cfg.overwrite
            raw_struct_curr = preprocessRCS(p_data_in_full_curr, str_session, ...
                str_device_curr, cfg);
            save(fullfile(p_data_out_raw_curr, f_raw_struct_curr), 'raw_struct_curr')
        else
            fprintf("Loading %s\n", fullfile(p_data_out_raw_curr, f_raw_struct_curr))
            load(fullfile(p_data_out_raw_curr, f_raw_struct_curr), 'raw_struct_curr');
        end
        vec_output{idx_session} = raw_struct_curr;


        % Visualizing the effect of removing stimulation artifacts
        plotRawComp(p_figure, raw_struct_curr, str_session, [], ...
            str_side, cfg.str_data_day, cfg)
        plotRawECoGComp(p_figure, raw_struct_curr, str_session, ...
            [], str_side, cfg.str_data_day, cfg)

    end

    % subsequently combine all sessions
    % loop through all the sessions
    pow_data_motor_full = [];
    t_pow_abs_motor_full = [];

    for idx_sess = 1:numel(vec_output)
        % unpack the variables
        output_curr = vec_output{idx_sess};

        pow_data_motor_curr = output_curr.pow_data_motor;
        t_pow_abs_motor_curr = output_curr.t_pow_abs;

        % now pack to outer structure
        pow_data_motor_full = [pow_data_motor_full; pow_data_motor_curr];
        t_pow_abs_motor_full = [t_pow_abs_motor_full; t_pow_abs_motor_curr];

    end

    % next sort the motor power data
    str_motor_band = output_curr.vec_str_pow_band_motor;
    [~, idx_arg_sort_pow_motor] = sort(t_pow_abs_motor_full);
    t_pow_abs_motor_sorted = t_pow_abs_motor_full(idx_arg_sort_pow_motor);
    pow_data_motor_sorted = pow_data_motor_full(idx_arg_sort_pow_motor, :);

    % now try to load in motor diary
    p_md = '/home/jyao/local/data/starrlab/raw_data/RCS08/Step6_at_home aDBS_short/06072022/motor diary/';
    f_md = 'RCS08_MotorDiary_06072022_formatted.xlsx';
    pf_md = fullfile(p_md, f_md);

    md_curr = GetMotorDiary(pf_md, '06/07/2022');
    md_range = [0, 10];
    sess_start = datetime('07-Jun-2022 08:00:00.000', ...
        'Format', t_pow_abs_motor_sorted.Format);
    sess_end = datetime('07-Jun-2022 20:30:00.000', ...
        'Format', t_pow_abs_motor_sorted.Format);

    %% AUC analysis
    % first analyze the right arm bradykinesia
    md_r_arm_brady = md_curr(:, ["time", 'RArmBrady']);
    threshold_r_arm_brady = 1;  % note that the threshold is right inclusive - meaning that C1 = >=range(1) < threshold; C2 = >= threshold <= range(2)
    
    % first obtain the time ranges
    [time_ranges_class1_r_arm_brady, time_ranges_class2_r_arm_brady] = ...
        obtainMotorDiaryTimeRange(md_r_arm_brady, threshold_r_arm_brady, ...
        'RArmBrady', md_range, sess_start, sess_end);
    
    % subsequently extract data based on these time ranges
    output_data_r_arm_brady_band3 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 3, ...
        time_ranges_class1_r_arm_brady, time_ranges_class2_r_arm_brady);

    % now compute the AUC
    [ca_r_arm_brady_band3, ~, ~, AUC_r_arm_brady_band3, ~] = ...
        KFoldCross(output_data_r_arm_brady_band3, ...
        1:length(output_data_r_arm_brady_band3), 10);

    % subsequently extract data based on these time ranges
    output_data_r_arm_brady_band4 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 4, ...
        time_ranges_class1_r_arm_brady, time_ranges_class2_r_arm_brady);

    % now compute the AUC
    [ca_r_arm_brady_band4, ~, ~, AUC_r_arm_brady_band4, ~] = ...
        KFoldCross(output_data_r_arm_brady_band4, ...
        1:length(output_data_r_arm_brady_band4), 10);
    chance_acc_r_arm_brady = size(output_data_r_arm_brady_band4{1}, 1) / ...
        (size(output_data_r_arm_brady_band4{1}, 1) + size(output_data_r_arm_brady_band4{2}, 1));

    % then analyze right leg bradykinesia
    md_r_leg_brady = md_curr(:, ["time", 'RLegBrady']);
    threshold_r_leg_brady = 1;

    % first obtain the time ranges
    [time_ranges_class1_r_leg_brady, time_ranges_class2_r_leg_brady] = ...
        obtainMotorDiaryTimeRange(md_r_leg_brady, threshold_r_leg_brady, ...
        'RLegBrady', md_range, sess_start, sess_end);
    
    % subsequently extract data based on these time ranges
    output_data_r_leg_brady_band3 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 3, ...
        time_ranges_class1_r_leg_brady, time_ranges_class2_r_leg_brady);

    % now compute the AUC
    [ca_r_leg_brady_band3, ~, ~, AUC_r_leg_brady_band3, ~] = ...
        KFoldCross(output_data_r_leg_brady_band3, ...
        1:length(output_data_r_leg_brady_band3), 10);

    % subsequently extract data based on these time ranges
    output_data_r_leg_brady_band4 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 4, ...
        time_ranges_class1_r_leg_brady, time_ranges_class2_r_leg_brady);

    % now compute the AUC
    [ca_r_leg_brady_band4, ~, ~, AUC_r_leg_brady_band4, ~] = ...
        KFoldCross(output_data_r_leg_brady_band4, ...
        1:length(output_data_r_leg_brady_band4), 10);
    chance_acc_r_leg_brady = size(output_data_r_leg_brady_band4{1}, 1) / ...
        (size(output_data_r_leg_brady_band4{1}, 1) + size(output_data_r_leg_brady_band4{2}, 1));

    % subsequently analyze right arm dyskinesia
    md_r_arm_dys = md_curr(:, ["time", 'RArmDyskinesia']);
    threshold_r_arm_dys = 1;

    % first obtain the time ranges
    [time_ranges_class1_r_arm_dys, time_ranges_class2_r_arm_dys] = ...
        obtainMotorDiaryTimeRange(md_r_arm_dys, threshold_r_arm_dys, ...
        'RArmDyskinesia', md_range, sess_start, sess_end);
    
    % subsequently extract band3 data based on these time ranges
    output_data_r_arm_dys_band3 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 3, ...
        time_ranges_class1_r_arm_dys, time_ranges_class2_r_arm_dys);

    % now compute the AUC using band3
    [ca_r_arm_dys_band3, ~, ~, AUC_r_arm_dys_band3, ~] = ...
        KFoldCross(output_data_r_arm_dys_band3, ...
        1:length(output_data_r_arm_dys_band3), 10);

    % subsequently extract band4 data based on these time ranges
    output_data_r_arm_dys_band4 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 4, ...
        time_ranges_class1_r_arm_dys, time_ranges_class2_r_arm_dys);

    % now compute the AUC using band4
    [ca_r_arm_dys_band4, ~, ~, AUC_r_arm_dys_band4, ~] = ...
        KFoldCross(output_data_r_arm_dys_band4, ...
        1:length(output_data_r_arm_dys_band4), 10);
    chance_acc_r_arm_dys = size(output_data_r_arm_dys_band4{1}, 1) / ...
        (size(output_data_r_arm_dys_band4{1}, 1) + size(output_data_r_arm_dys_band4{2}, 1));


    % subsequently analyze right leg dyskinesia
    md_r_leg_dys = md_curr(:, ["time", 'RLegDyskinesia']);
    threshold_r_leg_dys = 1;

    % first obtain the time ranges
    [time_ranges_class1_r_leg_dys, time_ranges_class2_r_leg_dys] = ...
        obtainMotorDiaryTimeRange(md_r_leg_dys, threshold_r_leg_dys, ...
        'RLegDyskinesia', md_range, sess_start, sess_end);
    
    % subsequently extract band3 data based on these time ranges
    output_data_r_leg_dys_band3 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 3, ...
        time_ranges_class1_r_leg_dys, time_ranges_class2_r_leg_dys);

    % now compute the AUC using band3
    [ca_r_leg_dys_band3, ~, ~, AUC_r_leg_dys_band3, ~] = ...
        KFoldCross(output_data_r_leg_dys_band3, ...
        1:length(output_data_r_leg_dys_band3), 10);

    % subsequently extract band4 data based on these time ranges
    output_data_r_leg_dys_band4 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 4, ...
        time_ranges_class1_r_leg_dys, time_ranges_class2_r_leg_dys);

    % now compute the AUC using band4
    [ca_r_leg_dys_band4, ~, ~, AUC_r_leg_dys_band4, ~] = ...
        KFoldCross(output_data_r_leg_dys_band4, ...
        1:length(output_data_r_leg_dys_band4), 10);
    chance_acc_r_leg_dys = size(output_data_r_leg_dys_band4{1}, 1) / ...
        (size(output_data_r_leg_dys_band4{1}, 1) + size(output_data_r_leg_dys_band4{2}, 1));

    % subsequently analyze more stim
    md_more_stim = md_curr(:, ["time", 'moreStim']);
    threshold_more_stim = 1;

    % first obtain the time ranges
    [time_ranges_class1_more_stim, time_ranges_class2_more_stim] = ...
        obtainMotorDiaryTimeRange(md_more_stim, threshold_more_stim, ...
        'moreStim', md_range, sess_start, sess_end);
    
    % subsequently extract band3 data based on these time ranges
    output_data_more_stim_band3 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 3, ...
        time_ranges_class1_more_stim, time_ranges_class2_more_stim);

    % now compute the AUC using band3
    [ca_more_stim_band3, ~, ~, AUC_more_stim_band3, ~] = ...
        KFoldCross(output_data_more_stim_band3, ...
        1:length(output_data_more_stim_band3), 10);

    % subsequently extract band4 data based on these time ranges
    output_data_more_stim_band4 = extractTimeRangeData(...
        t_pow_abs_motor_sorted, pow_data_motor_sorted, 4, ...
        time_ranges_class1_more_stim, time_ranges_class2_more_stim);

    % now compute the AUC using band4
    [ca_more_stim_band4, ~, ~, AUC_more_stim_band4, ~] = ...
        KFoldCross(output_data_more_stim_band4, ...
        1:length(output_data_more_stim_band4), 10);
    chance_acc_more_stim = size(output_data_more_stim_band4{1}, 1) / ...
        (size(output_data_more_stim_band4{1}, 1) + size(output_data_more_stim_band4{2}, 1));
    t1 = 1;

end