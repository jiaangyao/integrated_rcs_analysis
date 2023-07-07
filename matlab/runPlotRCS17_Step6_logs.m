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

cfg.str_sub = 'RCS17';
cfg.str_paradigm = getenv('STR_aDBS');
cfg.str_aDBS_paradigm = getenv('STR_STEP6');
cfg.idx_aDBS_paradigm = split(cfg.str_aDBS_paradigm, '_');
cfg.idx_aDBS_paradigm = cfg.idx_aDBS_paradigm{1};
cfg.idx_aDBS_paradigm = str2num(cfg.idx_aDBS_paradigm(end));

cfg.str_round = 'Round1';
cfg.str_data_day = '20230704';

% cfg.vec_str_side = {'Left', 'Right'};
cfg.vec_str_side = {'Left'};
% cfg.vec_str_side = {'Right'};
cfg.overwrite = false;
cfg.figure_overwrite = false;
cfg.thresCombo = false;
cfg.boolSaveAsFig = false;

cfg.str_no_pkg_data_day = {'20230605'};

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
    p_data_in_handle = fullfile(p_data_in, cfg.str_paradigm, cfg.str_sub, ...
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

        raw_struct_curr = preprocessRCSLogs(p_data_in_full_curr, str_session, ...
            str_device_curr, cfg);
        vec_output{idx_session} = raw_struct_curr;

    end
    
    % load the PKG data
    if ~any(strcmp(cfg.str_no_pkg_data_day, cfg.str_data_day))
        p_pkg = fullfile(p_data_in_handle, cfg.str_round, ...
            cfg.str_data_day, 'pkg', str_oppo_side(1));
        vec_pfe_pkg = glob(fullfile(p_pkg, 'scores*'));

        % sanity check - should only be one of these unless otherwise
        if numel(vec_pfe_pkg) > 1
            error("Should only be one of these");
        end
        
        % read and preprocess the PKG data table
        pkgWatchTable = readtable(vec_pfe_pkg{1});
        pkgWatchTable.Date_Time.TimeZone = vec_output{1}.metaData.timeZone;
        pkgWatchTable.Date_Time.Format = vec_output{1}.metaData.timeFormat;
        pkgWatchTable.BK = pkgWatchTable.BKS;
        pkgWatchTable.DK = pkgWatchTable.DKS;

        for i = 1:size(pkgWatchTable, 1)
            if strcmp(pkgWatchTable.Off_Wrist{i}, 'True')
                pkgWatchTable.Off_Wrist{i} = true;
            else
                pkgWatchTable.Off_Wrist{i} = false;
            end
        end

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
    plotLogsOverTime(pFigure, vec_output, pkgWatchTable, [], str_side, ...
        cfg.str_data_day, cfg, 'boolSaveAsFig', cfg.boolSaveAsFig)


end