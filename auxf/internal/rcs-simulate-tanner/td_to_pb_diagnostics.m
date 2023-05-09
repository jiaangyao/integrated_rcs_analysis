%% Function Name: td_to_pb_diagnostics()
%
% Description: Computes offline estimates of Power Band signals recorded on
% the RC+S device, then calculates several diagnostic measurements of the
% estimate performance and logs them alongside device settings.
%
% Inputs:
%     ui input : data directory path, selected through file explorer
%         Data directory must contain at least one session folder. Make
%         sure that only the JSON files are located in the session folders.
%         The JSON files may either be in the session folder itself, or in
%         a Device* subdirectory.
%     plot_results : optional boolean, default=false
%         Indicates whether to plot snippets of measured and estimated
%         Power Band data.
%     include_time_series : optional boolean, default=false
%         Indicates whether to output time series data in the diagnostics 
%         table. Set to `true` only if you do not mind the larger file
%         size, which will depend on the length of recordings.
%
% Outputs:
%     diagnostics : (n, 20) table
%         A table containing diagnostic outputs for n different blocks of
%         data. Each session folder will be logged as at least one row, and 
%         will be split into dedicated rows for each change in FFT or Power
%         settings during the session. The first six fields contain 
%         time-series data, which can be optionally included based on the 
%         input (default=false). The fields are:
%             time_td : (Optional) (n, 1) array of timestamps for the 
%                 Time-Domain data, in ms
%             data_td : (Optional) (n, 4) array of Time-Domain data, in mV.
%                 Columns represent different channels.
%             time_pb_true : (Optional) (n, 1) array of timestamps for the
%                 measured Power Bands, in ms.
%             data_pb_true : (Optional) (n, 8) array of Power Band data, in
%                 RC+S units. Columns represent different bands.
%             time_pb_est : (Optional) (n, 1) array of timestamps for the 
%                 estimates, in ms.
%             data_pb_est : (Optional) (n, 8) array of Offline Power Band 
%                 estimates, in Power Band RC+S units. Columns represent
%                 different bands.
%             fs_td : Time-Domain sampling rate, in Hz
%             L : FFT size, in samples
%             interval : Power Band sampling interval, in ms
%             bit_shift : bit shift parameter
%             window_load : Hann window load
%             amp_gain : (1,4) array containing channel amplifier gains
%             pb_bins : (8, 2) array containing indices of bin edges for 
%                 each of the eight recorded Power Bands
%             frechet : (1,8) array containing Frechet distance between the
%                 measured Power Bands and offline estimates
%             r : (1,8) array containing Pearson correlation coefficient  
%                 between measured Power Bands and offline estimates
%             R2 : (1,8) array containing R2 between measured Power Bands
%                 and offline estimates
%             RMSE : (1,8) array containing RMSE between measured Power  
%                 Bands and offline estimates
%             percent_error : (1,8) array containing percent error between  
%                 measured Power Bands and offline estimates
%             mean_shift : (1,8) array containing mean shift between 
%                 measured Power Bands and offline estimates
%             value_range : (1,8) array containing the range of values in 
%                 offline Power Band estimates, in RC+S units
%
% Author: Tanner Chas Dixon, tanner.dixon@ucsf.edu
% Date last updated: March 2, 2022
%---------------------------------------------------------

function [diagnostics] = td_to_pb_diagnostics(plot_results, ...
                                              include_time_series)

% Validate function arguments and set defaults
arguments
    plot_results {mustBeNumericOrLogical} = false
    include_time_series {mustBeNumericOrLogical} = false
end

% Select data folder 
disp('Select directory containing session folders:')
outer_dir = uigetdir();
session_list = dir(outer_dir);
session_list = session_list([session_list.isdir] ...
                            & ~ismember({session_list.name},{'.','..'}));

% Iterate over all sessions in the selected folder and fill the output
% table with diagnostic information
diagnostics  = cell2table(cell(0,20), 'VariableNames', ...
         {'time_td', 'data_td', ...
          'time_pb_true', 'data_pb_true', ...
          'time_pb_est', 'data_pb_est', ...
          'fs_td', 'L', 'interval', 'bit_shift', 'window_load', ...
          'amp_gain', 'pb_bins', 'frechet', 'r', 'R2' 'RMSE', ...
          'percent_error', 'mean_shift', 'value_range'});
for session_idx = 1:length(session_list)
    session_dir = session_list(session_idx).name;
    session_dir = fullfile(outer_dir, session_dir);
    % If the JSON files are embedded in a "device" subdirectory, access it
    if isempty(dir([session_dir, '*.json']))
        device_dir = dir(session_dir);
        device_dir = device_dir([device_dir.isdir] ...
                                & ~ismember({device_dir.name},{'.','..'}));
        session_dir = fullfile(session_dir, device_dir.name);
    end
    % Load the contents of the JSON files
    [~, timeDomainData, ~, ~, ~, ~, ~, PowerData, ~, ~, ~, ~, ~, ~, ~,...
     ~, timeDomainSettings, powerSettings, ~, ~, metaData, ~,...
     ~, ~, ~, ~, ~, ~] = ProcessRCS(session_dir, 2);

    % Iterate over recordings "blocks", where a block is defined as a
    % recording segment with consistent sense settings. Many recordings
    % will have only one block.
    for block = 1:size(powerSettings,1)

        % extract Time-Domain data and measured Power Band signals
        start_time = powerSettings.timeStart(block);
        stop_time = powerSettings.timeStop(block);
        if (stop_time - start_time) / 1000 < 30
            continue
        end
        td_mask = (timeDomainData.DerivedTime>=start_time) ...
                  & (timeDomainData.DerivedTime<=stop_time);
        time_td = timeDomainData.DerivedTime(td_mask);
        data_td = [timeDomainData.key0(td_mask),...
                   timeDomainData.key1(td_mask), ...
                   timeDomainData.key2(td_mask), ...
                   timeDomainData.key3(td_mask)];
        pb_mask = (PowerData.newDerivedTime>=start_time) ...
                  & (PowerData.newDerivedTime<=stop_time);
        pb_true = [PowerData.Band1(pb_mask),...
                   PowerData.Band2(pb_mask),...
                   PowerData.Band3(pb_mask),...
                   PowerData.Band4(pb_mask),...
                   PowerData.Band5(pb_mask),...
                   PowerData.Band6(pb_mask),...
                   PowerData.Band7(pb_mask),...
                   PowerData.Band8(pb_mask)];
        time_pb = PowerData.newDerivedTime(pb_mask);

        % Assign all the settings involved in estimating offline Power Bands
        fs_td = timeDomainSettings.TDsettings{block,1}(1).sampleRate;
        L = powerSettings.fftConfig(block,1).size;
        interval = powerSettings.fftConfig(block,1).interval;
        bit_shift = str2double(...
                 powerSettings.fftConfig(block,1).bandFormationConfig(6));
        win_load = powerSettings.fftConfig(block,1).windowLoad;
        hann_win = hannWindow(L, win_load);
        amp_gain = [metaData.ampGains.Amp1, metaData.ampGains.Amp2,...
            metaData.ampGains.Amp3, metaData.ampGains.Amp4];
        pb_bins = ...
              powerSettings.powerBands(block,1).indices_BandStart_BandStop;
        center_freqs = (0:(L/2-1)) * fs_td/L;

        % Compute offline estimates of all recorded power band signals
        pb_est = [];
        frechet = [];
        r = [];
        R2 = [];
        RMSE = [];
        percent_error = [];
        mean_shift = [];
        value_range = [];
        for k = 1:8
            % grab the time-domain data and amplifier gain
            if k<3 % amp1
                data_td_mv = data_td(:,1);
                current_amp_gain = amp_gain(:,1);
            elseif k<5 % amp2
                data_td_mv = data_td(:,2);
                current_amp_gain = amp_gain(:,2);
            elseif k<7 % amp3
                data_td_mv = data_td(:,3);
                current_amp_gain = amp_gain(:,3);
            else %amp4
                data_td_mv = data_td(:,4);
                current_amp_gain = amp_gain(:,4);
            end

            % select the frequency band
            pb_band_idx = pb_bins(k,:);
            band_edges_hz = center_freqs(pb_band_idx);

            % compute the estimate
            % TD to FFT
            data_td_rcs = transformMVtoRCS(data_td_mv, current_amp_gain);
            [data_fft, time_fft] = rcs_td_to_fft(data_td_rcs, time_td, ...
                fs_td, L, interval, hann_win);
            % FFT to PB
            pbX_est = rcs_fft_to_pb(data_fft, fs_td, L, bit_shift, ...
                band_edges_hz);
            pb_est = [pb_est, pbX_est];

            % Compute diagnostic metrics
            [pb_true_resampled, pb_est_resampled] ...
                = resample_pb(time_pb, pb_true(:,k), time_fft, pbX_est);
            frechet = [frechet; ...
                     z_score_frechet(pb_true_resampled, pb_est_resampled)];
            r = [r; corr(pb_true_resampled, pb_est_resampled)];
            R2 = [R2; 1 - sum((pb_true_resampled - pb_est_resampled).^2) ...
                / sum((pb_true_resampled - mean(pb_true_resampled)).^2)];
            RMSE = [RMSE; ...
                sqrt(mean((pb_true_resampled - pb_est_resampled).^2))];
            percent_error = [percent_error; ...
                mean((pb_est_resampled - pb_true_resampled)...
                ./ pb_true_resampled)];
            mean_shift = [mean_shift; ...
                mean(pb_est_resampled - pb_true_resampled)];
            value_range = [value_range; ...
                           [quantile(pb_est_resampled,0.25), ...
                            quantile(pb_est_resampled,0.75)]];

        end

        % Log diagnostic information in output table
        diagnostics = [diagnostics; ...
            {time_td, data_td, time_pb, pb_true, time_fft, pb_est,...
            fs_td, L, interval, bit_shift, win_load, amp_gain, pb_bins,...
            frechet, r, R2, RMSE, percent_error, mean_shift, value_range}];
    end


end

% Plot the data if requested
if plot_results
    for block = 1:size(diagnostics,1)
        plot_estimates(diagnostics(block,:), block)
    end
end

% Remove time-series unless requested
if ~include_time_series
    diagnostics= removevars(diagnostics,{'time_td', 'data_td',...
                                         'time_pb_true', 'data_pb_true',...
                                         'time_pb_est', 'data_pb_est'});
end

end


%% Helper functions

function [pb_true_resampled, pb_est_resampled] ...
                  = resample_pb(time_pb_true, pb_true, time_pb_est, pb_est)
% Resample estimates at the same timestamps as the true signals
pb_est_resampled = interp1(time_pb_est, pb_est, time_pb_true);
% Throw away the first ten seconds, since these are often aberrant
keep_mask = (time_pb_true-time_pb_true(1)) > (10*1000);
keep_mask(end) = false;
pb_true_resampled = pb_true(keep_mask);
pb_est_resampled = pb_est_resampled(keep_mask);
end


function [frechet] = z_score_frechet(pb_true, pb_est)
if any(isnan(pb_true))
    frechet = [nan, nan];
    return
end
% Compute the mean and standard deviation of the measured signal
mu = mean(pb_true);
sigma = std(pb_true);
% Standardize both the measured and estimated signals
pb_true = (pb_true - mu) / sigma;
pb_est = (pb_est - mu) / sigma;
pb_null = zeros(size(pb_true));
% Compute the Frechet distance in every 1000 sample chunk and take the
% average. This reduces computation time.
num_chunks = ceil(length(pb_true) / 1000);
frechet_chunks = zeros([1,num_chunks]);
for chunk = 1:num_chunks
    start = 1 + (num_chunks-1)*1000;
    stop = min([num_chunks*1000, length(pb_true)]);
    [current_frichet, ~] = DiscreteFrechetDist(pb_true(start:stop), ...
                                               pb_est(start:stop));
    frechet_chunks(chunk) = current_frichet;
end
if (stop - start) < 100
    frechet_chunks(end) = [];
end
frechet = mean(frechet_chunks);
% Include a null calculation where the measured signal is compared with its
% own mean
frechet_chunks = zeros([1,num_chunks]);
for chunk = 1:num_chunks
    start = 1 + (num_chunks-1)*1000;
    stop = min([num_chunks*1000, length(pb_true)]);
    [current_frichet, ~] = DiscreteFrechetDist(pb_true(start:stop), ...
                                               pb_null(start:stop));
    frechet_chunks(chunk) = current_frichet;
end
if (stop - start) < 100
    frechet_chunks(end) = [];
end
frechet = [mean(frechet_chunks), frechet];
end


function plot_estimates(block, block_idx)
figure('Name', ['Block: ', num2str(block_idx)], ...
       'Position',[100,50,1200,900])

plot_range = min([100000, ...
                  length(block.time_pb_true{1}),...
                  length(block.time_pb_est{1})]);
time_pb_true = block.time_pb_true{1}(1:plot_range);
data_pb_true = block.data_pb_true{1}(1:plot_range,:);
time_pb_est = block.time_pb_est{1}(1:plot_range);
data_pb_est = block.data_pb_est{1}(1:plot_range,:);

zero_time = time_pb_true(1);
center_freqs = (0:(block.L/2-1)) * block.fs_td/block.L;
bin_edges_hz = center_freqs(block.pb_bins{1});
for k = 1:8
    ttl = {['Power band ', num2str(k),] ...
           ['bin indices [',num2str(block.pb_bins{1}(k,:)),']'], ...
           ['bin frequencies [',num2str(bin_edges_hz(k,:)),'] Hz']};
    subplot(4,2,k)
    plot((time_pb_true-zero_time)/1000, ...
         data_pb_true(:,k), 'LineWidth',1)
    hold on
    plot((time_pb_est-zero_time)/1000, ...
         data_pb_est(:,k), 'LineWidth',1)
    grid on
    ylabel({'Power', '[RCS units]'})
    xlabel('Time [sec]')
    title(ttl)
    xlim([1,120])
    ylim([0, quantile(data_pb_est(:,k), 0.99)])
end
legend({'Measured', 'Computed from TD'})
end



