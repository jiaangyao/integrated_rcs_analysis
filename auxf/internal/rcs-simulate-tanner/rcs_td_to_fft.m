%% Function Name: rcs_td_to_fft()
%
% Description: Computes short-time FFT of Time-Domain data as it is
% performed onboard the RC+S device. The result may optionally be converted
% to match the logged FFT outputs in units of mV.
%
% Inputs:
%     data_td : (num_samples, 1) array, or transpose
%         Time-Domain data, given in internal RC+S units.
%     time_td : (num_samples, 1) array, or transpose
%         Unix timestamps (in whole ms format) for the corresponding
%         Time-Domain data samples.
%     fs_td : int
%         The Time-Domain sampling rate, in Hz.
%     L : int, {64, 256, 1024}
%         FFT size, in number of samples.
%     interval : int 
%         The interval, in ms, that the FFT window is shifted for producing
%         each subsequent output sample.
%     hann_win : (L,1) array, or transpose
%         The coefficients of the Hann window applied to Time-Domain data 
%         prior to computing the FFT.
%     output_in_mv : optional boolean, default=false
%         Boolean flag indicating whether to match the FFT output units
%         to what is logged by the device (scaled mV).
%
% Outputs:
%     data_fft : (num_windows, L) array
%         FFT amplitude data given in internal RC+S units, or converted to 
%         match the scaled mV units that the device outputs in data logs if
%         specified by the `output_in_mv` parameter.
%     time_fft : (num_windows, 1) array
%         Unix timestamps for the corresponding FFT data samples.
%
% Author: Tanner Chas Dixon, tanner.dixon@ucsf.edu. Credit to Juan Anso for
%             earlier version of the code.
% Date last updated: February 14, 2022
%---------------------------------------------------------

function [data_fft, time_fft] = rcs_td_to_fft(data_td, time_td, fs_td,...
                                       L, interval, hann_win, output_in_mv)

% Validate function arguments and set defaults
arguments
    data_td {mustBeNumeric}
    time_td {mustBeNumeric}
    fs_td {mustBeInteger}
    L {mustBeMember(L,[64,256,1024])} 
    interval {mustBeInteger}
    hann_win {mustBeNumeric}
    output_in_mv {mustBeNumericOrLogical} = false
end

% Make sure all vectors are given as column vectors. This is important for
% the Hann window operation
data_td = data_td(:);
time_td = time_td(:);
hann_win = hann_win(:);

% The actual FFT uses a smaller number of true time-domain samples and
% zero-pads the remainder
switch L
    case 64
        L_non_zero = 62; % on the device this alternates between 62 and 63
    case 256
        L_non_zero = 250;
    case 1024
        L_non_zero = 1000;
end

% Linearly interpolate over NaN-values... TO-DO, CONSIDER A DIFFERENT FIX
nan_mask = isnan(data_td);
idx = 1:numel(data_td);
data_td(nan_mask) = interp1(idx(~nan_mask), data_td(~nan_mask), ...
    idx(nan_mask));

% Pre-select all FFT window edges
mean_window_shift = interval*fs_td/1000;
num_windows = floor((length(data_td)-L)/mean_window_shift) + 1;
window_stops = ceil((0:num_windows-1)*mean_window_shift) + L_non_zero;
window_starts = window_stops - L_non_zero + 1;
time_fft = time_td(window_stops);
data_fft = zeros(num_windows, L);
% Iterate over FFT windows
for s = 1:num_windows
    % Select the time-domain window and zero-pad remaining points
    td_window = zeros(L,1);
    td_window(1:L_non_zero) = ...
        data_td(window_starts(s):window_stops(s));
    % Take the FFT and calculate complex magnitudes
    current_fft = fft(td_window.*hann_win, L);
    current_fft = abs(current_fft);
    data_fft(s,:) = current_fft;
end

% Convert the units to match the logged output, if desired
if output_in_mv
    data_fft = transformRCStoMV(4*data_fft/L);
end

end
