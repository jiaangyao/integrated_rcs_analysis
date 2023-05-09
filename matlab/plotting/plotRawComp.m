function plotRawComp(p_figure, output, str_session, str_on_off_meds, ...
    str_side, str_data_day, cfg)

% first generate the figure path and check if figure already exists
% and whether or not to overwrite
if isempty(str_on_off_meds) == 0
    if cfg.bool_in_clinic
        str_on_off_meds_full = sprintf('%s_Meds', upper(str_on_off_meds));
    else
        str_on_off_meds_full = sprintf('%s_stim', str_on_off_meds);
    end
end

if strcmp(str_side, 'Left')
    str_sub_side_full = sprintf('%s%s', cfg.str_sub, 'L');
elseif strcmp(str_side, 'Right')
    str_sub_side_full = sprintf('%s%s', cfg.str_sub, 'R');
end

% set the output folder path
if isempty(str_on_off_meds) == 0
    p_figure_out  = fullfile(p_figure, cfg.str_sub, cfg.str_paradigm, str_data_day, ...
        str_sub_side_full, str_on_off_meds_full, 'raw_comp');
    f_figure_out = sprintf("raw_data_comp_%s_%s.png", lower(str_on_off_meds), ...
        str_session);
else
    p_figure_out  = fullfile(p_figure, cfg.str_sub, cfg.str_paradigm, str_data_day, ...
        str_sub_side_full, 'raw_comp');
    f_figure_out = sprintf("raw_data_comp_%s.png", str_session);
end
if exist(p_figure_out, 'dir') ~= 7
    mkdir(p_figure_out);
end

if exist(fullfile(p_figure_out, f_figure_out), 'file') == 0 || cfg.figure_overwrite
    % plotting the data before and after
    time = output.time;
    raw_data = output.raw_data;
    vec_str_chan = output.metaData.timeDomainMetaData.vecStrChan;
    
    t_stn_wnan = output.t_stn_wnan;
    raw_data_stn_wnan = output.raw_data_stn_wnan;
    
    t_stn_wnan_corr = output.t_stn_wnan_corr;
    raw_data_stn_wnan_corr = output.raw_data_stn_wnan_corr;
    
    
    
    f_raw_comp = figure(); clf; subplot(311); hold on;
    plot(time, raw_data(:, 1), "DisplayName", vec_str_chan{1});
    plot(time, raw_data(:, 2), "DisplayName", vec_str_chan{2});
    title("Raw data before artifact removal")
    ylabel("Amplitude (\muV)")
    legend;
    
    subplot(312); hold on;
    plot(t_stn_wnan, raw_data_stn_wnan(:, 1), "DisplayName", vec_str_chan{1});
    plot(t_stn_wnan, raw_data_stn_wnan(:, 2), "DisplayName", vec_str_chan{2});
    title("Raw data after artifact removal")
    ylabel("Amplitude (\muV)")
    subplot(313); hold on;
    plot(t_stn_wnan_corr, raw_data_stn_wnan_corr(:, 1), "DisplayName", vec_str_chan{1});
    plot(t_stn_wnan_corr, raw_data_stn_wnan_corr(:, 2), "DisplayName", vec_str_chan{2});
    title("Raw data after removing isolated chunks")
    xlabel("Time (s)")
    ylabel("Amplitude (\muV)")
    
    % now save the figure
    set(f_raw_comp, 'Position', [10, 10, 600, 1000]);
    saveas(f_raw_comp, fullfile(p_figure_out, f_figure_out));
    close(f_raw_comp)

end

end