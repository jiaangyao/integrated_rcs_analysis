%% Function Name: rcs_fft_to_pb()
%
% Description: Converts short-time FFT outputs to scaled Power Band signals 
% (or full spectrogram) with the same scaling operations performed onboard 
% the RC+S device.
%
% Inputs:
%     data_fft : (num_windows, L) array
%         FFT amplitude data given in internal RC+S units. This may also be
%         given in units of mV (matching the format in FFT data logs) if 
%         specified by the `input_is_mv` parameter.
%     fs_td : int
%         The Time-Domain sampling rate, in Hz.
%     L : int, {64, 256, 1024}
%         FFT size, in number of samples.
%     bit_shift : int, 0:7
%         Parameter indicating the number of most-significant-bits to be
%         discarded. This value should be input as exactly the same value
%         programmed on the device.
%     band_edges_hz : optional (num_bands, 2) array, default=[]
%         Edges of each power band requested, in Hz. If empty, the function
%         will return the full L/2-dimensional single-sided spectrogram.
%     input_is_mv : optional boolean, default=false
%         Boolean flag indicating whether the FFT input was given in units
%         of scaled mV, matching the format in the raw data logs.
%
% Outputs:
%     data_pb : (num_windows, num_bands) array
%         Power Band data given in internal RC+S units, or the full
%         L/2-dimensional spectrogram. Note that the first bin (DC) may be
%         double what it should be.
%
% Author: Tanner Chas Dixon, tanner.dixon@ucsf.edu. Credit to Juan Anso for
%             earlier version of the code.
% Date last updated: February 14, 2022
%---------------------------------------------------------

function data_pb = rcs_fft_to_pb(data_fft, fs_td, L, bit_shift, ...
                                 band_edges_hz, input_is_mv)

% Validate function arguments and set defaults
arguments
    data_fft {mustBeNumeric}
    fs_td {mustBeInteger}
    L {mustBeMember(L,[64,256,1024])} 
    bit_shift {mustBeMember(bit_shift,0:7)} 
    band_edges_hz {mustBeNumeric} = []
    input_is_mv {mustBeNumericOrLogical} = false
end

% If `data_fft` was given in mV, convert back to internal RCS units
if input_is_mv
    data_fft = transformMVtoRCS(L*data_fft/4);
end
% Convert amplitude to single-sided power spectrum
data_fft = data_fft.^2;
data_fft = 64 * data_fft(:,1:L/2) / (L^2); % all scaling collapsed into 64
data_fft(:,1) = data_fft(:,1)/2; % first bin (DC) adjustment
% Perform the bit-shift
data_pb = floor(data_fft/(2^(8-bit_shift))); % TO-DO: HANDLE OVERFLOW

% Sum over the bins in each power band or return the full spectrum if none
% given
if ~isempty(band_edges_hz)
    % Create a vector containing the center frequencies of all FFT bins
    center_freqs = (0:(L/2-1)) * fs_td/L;
    % For each requested band, sum over the appropriate FFT bins
    data_pb_binned = zeros(size(data_pb,1), size(band_edges_hz,1));
    for band_idx = 1:size(band_edges_hz,1)
        bin_mask = (center_freqs>=band_edges_hz(band_idx,1)) ...
                   & (center_freqs<=band_edges_hz(band_idx,2));
               bin_mask(11) = 0;
        data_pb_binned(:,band_idx) = sum(data_pb(:,bin_mask),2);
    end
    data_pb = data_pb_binned;
end

end



