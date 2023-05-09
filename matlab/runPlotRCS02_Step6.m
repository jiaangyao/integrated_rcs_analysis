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
cfg.str_aDBS_paradigm = getenv('STR_STEP6');
cfg.idx_aDBS_paradigm = split(cfg.str_aDBS_paradigm, '_');
cfg.idx_aDBS_paradigm = cfg.idx_aDBS_paradigm{1};
cfg.idx_aDBS_paradigm = str2num(cfg.idx_aDBS_paradigm(end));
% cfg.str_round = 'first round';
% cfg.str_data_day = '20220912';
% cfg.str_data_day = '20220914';
% cfg.str_data_day = '20220916';
% cfg.str_data_day = '20220918';
% cfg.str_data_day = '20220920';

% cfg.str_round = 'second round';
% cfg.str_data_day = '20220922';
% cfg.str_data_day = '20220925';
% cfg.str_data_day = '20220927';
% cfg.str_data_day = '20220928';

% cfg.str_round = 'third round';
% cfg.str_data_day = '20221024';
% cfg.str_data_day = '20221026';
% cfg.str_data_day = '20221028';
% cfg.str_data_day = '20221030';
% cfg.str_data_day = '20221101';

% cfg.str_round = 'fourth round';
% cfg.str_data_day = '20221114AM';
% cfg.str_data_day = '20221114PM';
% cfg.str_data_day = '20221116AM';
% cfg.str_data_day = '20221116PM';
% cfg.str_data_day = '20221118AM';
% cfg.str_data_day = '20221118PM';
% cfg.str_data_day = '20221120';
% cfg.str_data_day = '20221121';
% cfg.str_data_day = '20221123AM';
% cfg.str_data_day = '20221123PM';

cfg.str_round = 'fifth round';
% cfg.str_data_day = '20221130AM';
% cfg.str_data_day = '20221130PM';
% cfg.str_data_day = '20221202AM';
% cfg.str_data_day = '20221202PM';
% cfg.str_data_day = '20221204';
% cfg.str_data_day = '20221205';
% cfg.str_data_day = '20221208';
cfg.str_data_day = '20221209';

% cfg.vec_str_side = {'Left'};
cfg.vec_str_side = {'Right'};
cfg.overwrite = false;
cfg.figure_overwrite = false;
cfg.thresCombo = false;
cfg.boolSaveAsFig = true;

cfg.str_no_pkg_data_day = {'20221024', '20221026', '20221028', ...
    '20221030', '20221101', '20221114AM', '20221114PM', '20221116AM', ...
    '20221116PM', '20221118AM', '20221118PM', '20221120', '20221121', ...
    '20221123AM', '20221123PM', '20221130AM', '20221130PM', '20221202AM', ...
    '20221202PM', '20221204', '20221205', '20221208', '20221209'};
cfg.str_no_aw_data_day = {'20220925', '20221024', '20221026', ...
    '20221028', '20221030', '20221101', '20221114AM', '20221114PM', ...
    '20221116AM', '20221116PM', '20221118AM', '20221118PM', '20221120', ...
    '20221121', '20221123AM', '20221123PM', '20221130AM', '20221130PM', ...
    '20221202AM', '20221202PM', '20221204', '20221205', '20221208', ...
    '20221209'};
cfg.str_combo_w_LD1 = {'20221030', '20221114AM', '20221114PM', '20221116AM', ...
    '20221116PM', '20221118AM', '20221118PM', '20221120', '20221121', ...
    '20221123AM', '20221123PM', '20221130AM', '20221130PM', '20221202AM', ...
    '20221202PM', '20221204', '20221205', '20221208', '20221209'};

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

% step 6 specific parameters
if cfg.idx_aDBS_paradigm == 6
    cfg.analyze_step6 = true;
else
    cfg.analyze_step6 = false;
end

cfg = getDeviceName(cfg);


%% Setting analysis related hyperparameters


cfg.len_epoch_prune = 2;                                                    % length of epoch in seconds used for artifact removal
cfg.amp_thres = 20000;                                                      % amplitude threshold of STN artifacts
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
        str_oppo_side = 'Right';
    elseif strcmp(str_side, 'Right')
        str_sub_side_full = sprintf('%s%s', cfg.str_sub, 'R');
        str_device_curr = cfg.str_device_R;
        str_oppo_side = 'Left';
    else
        error('invalid side');
    end

    % now obtain the full input path and get ready for glob
    % first set the handle and then adjust path based on which step it is
    p_data_in_handle = fullfile(p_data_in, cfg.str_sub, cfg.str_paradigm, ...
        cfg.str_aDBS_paradigm);
    if cfg.analyze_step3
        error("Not implemented yet")
    elseif cfg.analyze_step4
        error("Not implemented yet")
    elseif cfg.analyze_step5
        p_data_in_full_curr = fullfile(p_data_in_handle, cfg.str_data_day, ...
            cfg.str_data_type, ...
            cfg.str_data_folder, str_sub_side_full);
        
        % set the output directory also
        p_data_out_full_curr = fullfile(p_data_out, cfg.str_sub, ...
            cfg.str_paradigm, cfg.str_aDBS_paradigm, ...
            cfg.str_data_day, str_sub_side_full);

    elseif cfg.analyze_step6
        p_data_in_full_curr = fullfile(p_data_in_handle, ...
            cfg.str_round, cfg.str_data_day, cfg.str_data_type, ...
            cfg.str_data_folder, str_sub_side_full);

        % set the output directory also
        p_data_out_full_curr = fullfile(p_data_out, cfg.str_sub, ...
            cfg.str_paradigm, cfg.str_aDBS_paradigm, ...
            cfg.str_round, cfg.str_data_day, str_sub_side_full);
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
            raw_struct_curr = SPFull_preprocRCS(p_data_in_full_curr, str_session, ...
                str_device_curr, cfg);
            save(fullfile(p_data_out_raw_curr, f_raw_struct_curr), ...
                'raw_struct_curr', '-v7.3')
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

    % now generate the power spectrum
    fprintf("\nProcessing power spectrum\n")
    [vec_pxx, vec_f, vec_str_chan_full_curr] = ...
        plotPSD(p_figure, vec_output, vec_str_session_curr, ...
        [], str_side, cfg.str_data_day, cfg, false, false);

    % now read in the motor diary
    pMD = {fullfile(p_data_in_handle, cfg.str_round, cfg.str_data_day, ...
        'motor diary', sprintf('%s_MotorDiary_%s_formatted.xlsx', ...
        cfg.str_sub, cfg.str_data_day))};
    dt = {strcat(cfg.str_data_day(5:6), '/', cfg.str_data_day(7:8), '/', ...
            cfg.str_data_day(1:4))};
    motorDiary = getMotorDiary(pMD, dt, ...
        raw_struct_curr.time_abs.Format, raw_struct_curr.time_abs.TimeZone);
    motorDiaryInterp = interpMotorDiary(motorDiary, cfg);

    if ~any(strcmp(cfg.str_no_aw_data_day, cfg.str_data_day))
        % now load the apple watch data
        p_apple_watch = fullfile(p_data_in_handle, cfg.str_round, ...
            cfg.str_data_day, 'Apple watch', str_oppo_side(1));

        table_dys_prob = readtable(fullfile(p_apple_watch, ...
            sprintf("%s_%s_DyskinesiaProb.csv", cfg.str_sub, ...
            str_oppo_side(1))));
        table_tre_prob = readtable(fullfile(p_apple_watch, ...
            sprintf("%s_%s_TremorProb.csv", cfg.str_sub, ...
            str_oppo_side(1))));

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
    else
        appleWatchTable = NaN;
    end

    if ~any(strcmp(cfg.str_no_pkg_data_day, cfg.str_data_day))
        p_pkg = fullfile(p_data_in_handle, cfg.str_round, ...
            'pkg', str_oppo_side(1));
        vec_pfe_pkg = glob(fullfile(p_pkg, 'scores*'));

        % sanity check - should only be one of these unless otherwise
        if numel(vec_pfe_pkg) > 1
            error("Should only be one of these");
        end
        
        % read and preprocess the PKG data table
        pkgWatchTable = readtable(vec_pfe_pkg{1});
        pkgWatchTable.Date_Time.TimeZone = vec_output{1}.time_abs.TimeZone;
        pkgWatchTable.Date_Time.Format = vec_output{1}.time_abs.Format;
        pkgWatchTable.BK = -pkgWatchTable.BK;

    else
        pkgWatchTable = NaN;
    end
    
    strSubSide = sprintf('%s%s', cfg.str_sub, cfg.vec_str_side{idx_side}(1));
    if cfg.boolSaveAsFig
        strFolderName = sprintf('TDAnalysis_%s_Step6_FIG', cfg.str_sub);
    else
        strFolderName = sprintf('TDAnalysis_%s_Step6', cfg.str_sub);
    end

    pFigure = fullfile('/home/jyao/Downloads/TDAnalysis', ...
        strFolderName, cfg.str_data_day, strSubSide);

    if ~exist(pFigure, 'dir')
        mkdir(pFigure);
    end

    % now plot the various features over time
    plotFeaturesOverTime(pFigure, vec_output, appleWatchTable, pkgWatchTable, ...
        motorDiaryInterp, motorDiary, [], str_side, cfg.str_data_day, cfg, ...
        'boolSaveAsFig', cfg.boolSaveAsFig)


end