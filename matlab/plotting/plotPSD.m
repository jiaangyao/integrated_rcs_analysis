function [vec_pxx, vec_f, vec_str_chan_full] = ...
    plotPSD(p_figure, vec_output, vec_str_session_curr, str_on_off_meds, ...
    str_side, str_data_day, cfg, bool_use_filt, bool_use_reref)

vec_pxx = {};
vec_f = {};
vec_str_chan_full = {};

% loop through all output files
for i = 1:numel(vec_output)
    output_curr = vec_output{i};

    % first generate the figure path and check if figure already exists
    % and whether or not to overwrite
    if strcmp(str_side, 'Left')
        str_sub_side_full = sprintf('%s%s', cfg.str_sub, 'L');
    elseif strcmp(str_side, 'Right')
        str_sub_side_full = sprintf('%s%s', cfg.str_sub, 'R');
    end

    
    if isempty(str_on_off_meds) == 0
        if cfg.bool_in_clinic
            str_on_off_meds_full = sprintf('%s_Meds', upper(str_on_off_meds));
        else
            str_on_off_meds_full = sprintf('%s_stim', str_on_off_meds);
        end
    end

    % set the output folder path
    if isempty(str_on_off_meds) == 0
        p_figure_out  = fullfile(p_figure, cfg.str_sub, cfg.str_paradigm, str_data_day, ...
            str_sub_side_full, str_on_off_meds_full, 'psd_all_ch');
    else
        p_figure_out  = fullfile(p_figure, cfg.str_sub, cfg.str_paradigm, str_data_day, ...
            str_sub_side_full, 'psd_all_ch');
    end
    if exist(p_figure_out, 'dir') ~= 7
        mkdir(p_figure_out);
    end

    f_figure_out = sprintf("psd_all_ch_%s", vec_str_session_curr(i).name);
    if bool_use_filt
        f_figure_out = sprintf('%s_filt', f_figure_out);
    else
        f_figure_out = sprintf('%s_no_filt', f_figure_out);
    end

    if bool_use_reref
        f_figure_out = sprintf('%s_reref.png', f_figure_out);
    else
        f_figure_out = sprintf('%s.png', f_figure_out);
    end

    bool_figure_exists = exist(fullfile(p_figure_out, f_figure_out), 'file') ~= 0 ...
        && ~cfg.figure_overwrite;

    % TODO: also optionally generate the PSD files
    
    % if file already exists and not overwriting figures
    if bool_figure_exists
        continue
    end
    
    % otherwise if no figure exists proceed and plot the figure
    fs_curr = output_curr.fs;
    if bool_use_filt
        raw_data_curr = output_curr.raw_data_filt;
    else
        raw_data_curr = output_curr.raw_data;
    end
    
    % optionally apply reref
    if bool_use_reref
        raw_data_curr = raw_data_curr - mean(raw_data_curr(:, 3:4), 2);
    end

    vec_str_chan_curr = output_curr.vec_str_chan;
    
    if bool_use_filt
        raw_data_stn_wnan_corr_curr = output_curr.raw_data_stn_wnan_filt_corr;
        t_stn_wnan_corr_curr = output_curr.t_stn_wnan_filt_corr;
    else
        raw_data_stn_wnan_corr_curr = output_curr.raw_data_stn_wnan_corr;
        t_stn_wnan_corr_curr = output_curr.t_stn_wnan_corr;
    end

    % optionally apply reref
    if bool_use_reref
        raw_data_stn_wnan_corr_curr = raw_data_stn_wnan_corr_curr - ...
            mean(raw_data_stn_wnan_corr_curr, 2);
    end

    idx_valid_abs_curr = ~isnan(t_stn_wnan_corr_curr);

    % compute the PSD for the STN channels
    [pxx_stn_chan1_curr, f_stn_chan1_curr] = ...
        pwelch(raw_data_stn_wnan_corr_curr(idx_valid_abs_curr(:, 1), 1), ...
        2^(nextpow2(fs_curr)), [], [], fs_curr);

    [pxx_stn_chan2_curr, f_stn_chan2_curr] = ...
        pwelch(raw_data_stn_wnan_corr_curr(idx_valid_abs_curr(:, 1), 2), ...
        2^(nextpow2(fs_curr)), [], [], fs_curr);
    
    % compute the PSD for the motor channels
    [pxx_motor_chan1_curr, f_motor_chan1_curr] = pwelch(...
        raw_data_curr(:, 3), ...
        2^(nextpow2(fs_curr)), [], [], fs_curr);
    [pxx_motor_chan2_curr, f_motor_chan2_curr] = pwelch(...
        raw_data_curr(:, 4), ...
        2^(nextpow2(fs_curr)), [], [], fs_curr);

    pxx_curr = [pxx_stn_chan1_curr, pxx_stn_chan2_curr, ...
        pxx_motor_chan1_curr, pxx_motor_chan2_curr];
    f_curr = [f_stn_chan1_curr, f_stn_chan2_curr, ...
        f_motor_chan1_curr, f_motor_chan2_curr];
    
    % now append to outer structure
    vec_pxx{i} = pxx_curr;
    vec_f{i} = f_curr;
    vec_str_chan_full{i} = vec_str_chan_curr;
    
    f_psd_curr = figure(); clf; 
    % get ready for plotting
    for idx_ch = 1:numel(vec_str_chan_curr)
        subplot(2, 2, idx_ch); hold on;
        str_ch_curr = vec_str_chan_curr{idx_ch};
        if idx_ch <= 2
            idx_valid_f = f_curr(:, idx_ch) >= cfg.lpf_edge_stn;
        else
            idx_valid_f = f_curr(:, idx_ch) >= cfg.lpf_edge_motor;
        end

        pxx_ch_curr = pxx_curr(:, idx_ch);
        f_ch_curr = f_curr(:, idx_ch);
        str_display_curr = sprintf('%s %s', str_on_off_meds, str_ch_curr);
        semilogy(f_ch_curr(idx_valid_f), log10(pxx_ch_curr(idx_valid_f)), ...
            'LineWidth', 2);
        
        if cfg.bool_old_recording
            xlim([0, 150])
        else
            xlim([0, 100])
        end
        ylim([-3, 3])

        xlabel("Frequency (Hz)")
        ylabel("Power (log_{10}\muV^2/Hz)")

        title(sprintf('%s %s %s %s %s', cfg.str_sub, strrep(str_data_day, "_", ' '), ...
            str_side, str_on_off_meds, str_ch_curr))

        beta_fill_x = [13, 30, 30, 13];
        fill_y = [-3, -3, 3, 3];
        fill(beta_fill_x, fill_y, 'g', 'EdgeColor', 'none', ...
            'FaceAlpha', 0.3);
    
        if cfg.bool_old_recording
            gamma_fill_x = [70, 150, 150, 70];
        else
            gamma_fill_x = [70, 100, 100, 70];
        end
        fill(gamma_fill_x, fill_y, 'y', 'EdgeColor', 'none', ...
            'FaceAlpha', 0.3);
    end

    % now save the figure
    sgtitle(vec_str_session_curr(i).name);
    set(f_psd_curr, 'Position', [10, 10, 1200, 1200]);
    saveas(f_psd_curr, fullfile(p_figure_out, f_figure_out));
    close(f_psd_curr)
end