function rawDataFilt = signalFiltBandpass(rawData, filtOrder, ...
    cutoffFreqLow, cutoffFreqHigh, fs)
%% signalUtils script - bandpass filters the signal
% Bandpass filters the input data using the MATLAB designfilt and FIR
% filters functionalities and user provided parameters
%
% INPUT:
% rawData               - double: raw data in format (time*channel)
% fs                    - double: sampling rate
% filtOrder             - double: order for filter
% cutoffFreqLow         - double: lower 6-dB frequency
% cutoffFreqHigh        - double: higher 6-dB frequency


%% sanity check

% check that array is in format (time*channel)
assert(size(rawData, 1) > size(rawData, 2), 'Data should be in format time*channel')


%% Quick test of filter order based on desired sharpness
% borrowed from EEGLAB code

% find Nyquist rate
fNyquist = fs / 2;

% define the filter edges
edgeArray = [cutoffFreqLow, cutoffFreqHigh];
if any(edgeArray < 0 | edgeArray >= fNyquist)
    error('Cutoff frequency out of range');
end

% check for filter order
if ~isempty(filtOrder) && (filtOrder < 2 || mod(filtOrder, 2) ~= 0)
    error('Filter order must be a real, even, positive integer.')
end

% Max stop-band width and modify for bandpass
maxTBWArray = edgeArray; % Band-/highpass
maxTBWArray(end) = fNyquist - edgeArray(end); % Band-/lowpass
maxDf = min(maxTBWArray);

% next figure out required df from Hamming window
filtOrderMin = ceil(3.3 ./ ((maxDf * 2) / fs) / 2) * 2;
filtOrderOpt = ceil(3.3 ./ (maxDf / fs) / 2) * 2;
if filtOrder < filtOrderMin
    error('Filter order too low. Minimum required filter order is %d.\nFor better results a minimum filter order of %d is recommended.', filtOrderMin, filtOrderOpt)
elseif filtOrder < filtOrderOpt
    warning('firfilt:filterOrderLow', 'Transition band is wider than maximum stop-band width. For better results a minimum filter order of %d is recommended. Reported might deviate from effective -6dB cutoff frequency.', filtOrderOpt)
end


%% filters the raw data

% design the filter first with the Kaiser window
bpfFilt = designfilt('bandpassfir', 'FilterOrder', filtOrder, ...
    	'CutoffFrequency1', cutoffFreqLow, 'CutoffFrequency2', cutoffFreqHigh,...
        'DesignMethod', 'window', 'Window', {@kaiser, 2.5}, 'SampleRate', fs);

% now perform zero-phase filtering
rawDataFilt = filtfilt(bpfFilt, rawData);


end