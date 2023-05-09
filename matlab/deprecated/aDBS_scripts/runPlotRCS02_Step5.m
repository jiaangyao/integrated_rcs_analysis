%% Envrionment setup

clearvars -except nothing;
clc;
close all;

set_env;
addpath(fullfile(getenv('D_ANALYZE_RCS'), 'code'));
addpath(genpath(fullfile('..', 'auxf', 'external')));
addpath(genpath(fullfile('..', 'auxf', 'internal/')));

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
cfg.str_sub = 'RCS02';
cfg.str_paradigm = getenv('STR_aDBS');
cfg.str_aDBS_paradigm = getenv('STR_STEP5');
cfg.idx_aDBS_paradigm = split(cfg.str_aDBS_paradigm, '_');
cfg.idx_aDBS_paradigm = cfg.idx_aDBS_paradigm{1};
cfg.idx_aDBS_paradigm = str2num(cfg.idx_aDBS_paradigm(end));
cfg.str_data_day = '05272022';
cfg.vec_str_side = {'Right'};
cfg.overwrite = true;
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
if cfg.idx_aDBS_paradigm == 3
    cfg.analyze_step3 = true;
else
    cfg.analyze_step3 = false;
end

% step 4 specific parameters
if cfg.idx_aDBS_paradigm == 4
    cfg.analyze_step4 = true;
else
    cfg.analyze_step4 = false;
end
cfg.vec_str_low_high_stim = {'low'};
cfg.bool_in_clinic = false;

% step 5 specific parameters
if cfg.idx_aDBS_paradigm == 5
    cfg.analyze_step5 = true;
else
    cfg.analyze_step5 = false;
end

cfg = getDeviceName(cfg);


%% Setting analysis related hyperparameters


cfg.len_epoch_prune = 2;                                                    % length of epoch in seconds used for artifact removal
if strcmp(cfg.str_sub, 'RCS08')
    cfg.amp_thres = 500;                                                    % amplitude threshold of STN artifacts
elseif strcmp(cfg.str_sub, 'RCS02')
    cfg.amp_thres = 100;
    cfg.amp_thres = 100;
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
        cfg.str_aDBS_paradigm, cfg.str_data_day);
    if cfg.analyze_step3
        error("Not implemented yet")
    elseif cfg.analyze_step4
        error("Not implemented yet")
    elseif cfg.analyze_step5
        p_data_in_full_curr = fullfile(p_data_in_handle, cfg.str_data_type, ...
            cfg.str_data_folder, str_sub_side_full);
        
        % set the output directory also
        p_data_out_full_curr = fullfile(p_data_out, cfg.str_sub, ...
            cfg.str_paradigm, cfg.str_aDBS_paradigm, ...
            cfg.str_data_day, str_sub_side_full);

    elseif cfg.analyze_step6
        p_data_in_full_curr = fullfile(p_data_in_handle, cfg.str_data_type, ...
            cfg.str_data_folder, str_sub_side_full);

        % set the output directory also
        p_data_out_full_curr = fullfile(p_data_out, cfg.str_sub, ...
            cfg.str_paradigm, cfg.str_aDBS_paradigm, ...
            cfg.str_data_day, str_sub_side_full);
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
    
        try
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
        catch
            t1 = 1;
        end

    end

    % now generate the power spectrum
    fprintf("\nProcessing power spectrum\n")
    [vec_pxx, vec_f, vec_str_chan_full_curr] = ...
        plotPSD(p_figure, vec_output, vec_str_session_curr, ...
        [], str_side, cfg.str_data_day, cfg, false, false);

    % now load the apple watch data
    p_apple_watch = '/home/jyao/local/data/starrlab/raw_data/RCS02/Step5_supervised_aDBS/05272022/Apple watch/L/';
    table_dys_prob = readtable(fullfile(p_apple_watch, "RCS02_L_DyskinesiaProb.csv"));
    table_tre_prob = readtable(fullfile(p_apple_watch, "RCS02_L_TremorProb.csv"));
    
    t_dys_unix_full = table2array(table_dys_prob(:, "time"));
    t_dys_full = datetime(t_dys_unix_full, 'ConvertFrom','epochtime',...
        'Format', vec_output{1}.time_abs.Format, 'TimeZone', 'UTC');
    t_dys_full.TimeZone = vec_output{1}.time_abs.TimeZone;
    prob_dys_full = table2array(table_dys_prob(:, "probability"));

    t_tre_unix_full = table2array(table_tre_prob(:, "time"));
    t_tre_full = datetime(t_tre_unix_full, 'ConvertFrom','epochtime',...
        'Format', vec_output{1}.time_abs.Format, 'TimeZone', 'UTC');
    t_tre_full.TimeZone = vec_output{1}.time_abs.TimeZone;
    prob_tre_full = table2array(table_tre_prob(:, "probability"));

    appleWatchTable.t_dys_full = t_dys_full;
    appleWatchTable.prob_dys_full = prob_dys_full;

    appleWatchTable.t_tre_full = t_tre_full;
    appleWatchTable.prob_tre_full = prob_tre_full;

    % now plot the various features over time
    plotFeaturesOverTime(p_figure, vec_output, appleWatchTable, ...
        [], str_side, cfg.str_data_day, cfg)


end