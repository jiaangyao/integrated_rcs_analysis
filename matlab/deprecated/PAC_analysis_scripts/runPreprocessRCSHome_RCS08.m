%% Envrionment setup

clearvars -except nothing;
clc;
close all;

set_env;
addpath(fullfile('..', 'analyze_rcs_data', 'code'));
addpath(fullfile('..', 'auxf', 'external'));

addpath(fullfile('preproc'))
addpath(fullfile('utils'))
addpath(fullfile('plotting'))


%% Setting the aDBS path

p_data_in = getenv('D_DATA_IN');
p_figure = getenv('D_FIGURE');
cfg.str_sub = 'RCS08';
cfg.str_paradigm = getenv('STR_ATHOME');
cfg.vec_str_data_day = {'01252022'};
cfg.str_data_type = getenv('STR_DATA_TYPE');
cfg.str_data_folder = getenv('STR_DATA_FOLDER');
cfg.vec_str_low_high_stim = {'low'};
cfg.vec_str_side = {'Left', 'Right'};
cfg.bool_in_clinic = false;

cfg.vec_str_old_recordings = {'10day'};

if any(strcmp(cfg.vec_str_old_recordings, cfg.vec_str_data_day))
    cfg.bool_old_recording = true;
else
    cfg.bool_old_recording = false;
end


cfg = getDeviceName(cfg);

cfg.len_epoch_prune = 2;
if strcmp(cfg.str_sub, 'RCS08')
    cfg.amp_thres = 500;
elseif strcmp(cfg.str_sub, 'RCS02')
    cfg.amp_thres = 100;
end

cfg.vec_epoch_pattern = {[0, 1, 0], [0, 0, 1, 1, 0]};
cfg.bool_apply_lpf_stn = true;
cfg.bool_apply_notch_stn = false;
cfg.lpf_edge_stn = 0.5;

cfg.bool_apply_lpf_motor = true;
cfg.bool_apply_notch_motor = false;
cfg.lpf_edge_motor = 0.5;

cfg.str_reref_mode = 'CAR';


cfg.bool_clim = true;
if strcmp(cfg.str_sub, 'RCS08')
        cfg.clim2 = 0.0002;
elseif strcmp(cfg.str_sub, 'RCS02')
    cfg.clim2 = 0.0002;
end
cfg.clim1 = 0;

% set the start of the recording to allow for system to settle...
if strcmp(cfg.str_sub, 'RCS08')
    cfg.t_start = 100;
    cfg.t_end = 220;
elseif strcmp(cfg.str_sub, 'RCS02')
    % cfg.t_start = 200;
    cfg.t_start = 100;
    cfg.t_end = 220;
end

%% Running the main loop

for idx_side = 1:length(cfg.vec_str_side)
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

    for idx_low_high_stim = 1:length(cfg.vec_str_low_high_stim)
        % obtain the directory name corresponding to current on/off med
        % paradigm
        str_low_high_stim = cfg.vec_str_low_high_stim{idx_low_high_stim};
        str_low_high_stim_full = sprintf('%s_stim', str_low_high_stim);
        str_data_day = cfg.vec_str_data_day{idx_low_high_stim};

        % now obtain the full path and get ready for glob
        p_data_in_full_curr = fullfile(p_data_in, cfg.str_sub, cfg.str_paradigm, ...
            str_low_high_stim_full, str_data_day, cfg.str_data_type, ...
            cfg.str_data_folder, str_sub_side_full);
        vec_str_session_curr = dir(fullfile(p_data_in_full_curr, 'Session*'));

        % now loop through all available sessions
        for idx_session = 1:numel(vec_str_session_curr)
            str_session = vec_str_session_curr(idx_session).name;

            output_curr = preprocessRCS(p_data_in_full_curr, str_session, ...
                str_device_curr, cfg);
            vec_output{idx_session} = output_curr;

            % Visualizing the effect of removing stimulation artifacts
            plotRawComp(p_figure, output_curr, str_session, str_low_high_stim, ...
                str_side, str_data_day, cfg)
            plotRawECoGComp(p_figure, output_curr, str_session, ...
                str_low_high_stim, str_side, str_data_day, cfg)

        end

        % now plot the current power spectral density
        % [vec_pxx_filt, vec_f_filt, vec_str_chan_full_curr] = ...
        %     plotPSD(p_figure, vec_output, vec_str_session_curr, ...
        %     str_on_off_meds, str_side, cfg, true, true);

        [vec_pxx, vec_f, vec_str_chan_full_curr] = ...
            plotPSD(p_figure, vec_output, vec_str_session_curr, ...
            str_low_high_stim, str_side, str_data_day, cfg, false, false);

        % vec_pxx_filt_full{idx_on_off_meds} = vec_pxx;
        % vec_f_filt_full{idx_on_off_meds} = vec_f;
        % vec_str_chan_full{idx_on_off_meds} = vec_str_chan_full_curr;

        vec_pxx_full{idx_low_high_stim} = vec_pxx;
        vec_f_full{idx_low_high_stim} = vec_f;
        vec_str_chan_full{idx_low_high_stim} = vec_str_chan_full_curr;

        % Now compute the phase amplitude coupling
        % plotComodulogram(p_figure, vec_output, vec_str_session_curr, ...
        %     str_on_off_meds, str_side, cfg, true, true, cfg.str_reref_mode)

        plotComodulogram(p_figure, vec_output, vec_str_session_curr, ...
            str_low_high_stim, str_side, str_data_day, cfg, ...
            true, false, cfg.str_reref_mode)

    end

    if length(cfg.vec_str_low_high_stim) >= 2
        % now plot the average PSD also
        % plotPSDAvgComp(p_figure, vec_pxx_filt_full, vec_f_filt_full, vec_str_chan_full, ...
        %     str_side, cfg, true)
    
        plotPSDAvgComp(p_figure, vec_pxx_full, vec_f_full, vec_str_chan_full, ...
            str_side, str_data_day, cfg, false)
    end
end

