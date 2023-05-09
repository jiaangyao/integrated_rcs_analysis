function metaDataStructOut = preprocLoadMetaDataFFT(fftData, fftSettings, ...
    metaDataIn)
%% Preprocessing wrapper script - loading of relevant FFT metadata
% Loads metadata pertaining to the FFT domain data such as window overlap
% and whether new samples are integer multiples
% 
% INPUT:
% fftData               - table: FFT domain data table from processRCS
% fftSettings           - table: FFT domain setting table from processRCS
% metaDataIn            - struct: most general metadata struct


%% Check for whether FFT setting is enabled

% check for whether or not these settings are enabled
if size(fftSettings, 1) > 1
    metaDataStructOut.isEnabled = true;
else
    metaDataStructOut.isEnabled = false;
end


%% Parses relevant FFT information 

% in either case can parse some information from FFT config
fftConfig = fftSettings{1, 'fftConfig'};
metaDataStructOut.intervalNSamples = metaDataIn.timeDomainMetaData.fs / ...
    (1/(1e-3 * fftConfig.interval));
metaDataStructOut.boolInExactFs = ...
    isinteger(metaDataStructOut.intervalNSamples);

% in cases where it's not integer then hardcode to see if sampling rate and
% FFT size are encountered
if ~metaDataStructOut.boolInExactFs
    if ~(metaDataIn.timeDomainMetaData.fs == 250 & fftConfig.size == 256)
        error("Unseen sampling rate and interavl, double check sampling rate")
    
    % otherwise hardcode the power sampling strategy
    else
        metaDataStructOut.vecFFTWinSample = [13; 12];
        metaDataStructOut.vecFFTWin = metaDataStructOut.vecFFTWinSample * ...
            1/metaDataIn.timeDomainMetaData.fs;
        metaDataStructOut.fs = 1/(1e-3 * fftConfig.interval);
    end
end


end