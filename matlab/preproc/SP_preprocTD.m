function timeDomainDataStructOut = SP_preprocTD(timeDomainData, ...
    metaDataIn, preprocCfg)
%% Preprocessing SP wrapper script - all processing of time domain data
% Processes time domain data from RCS
% 
% INPUT:
% timeDomainData              - table: time domain data table from processRCS
% metaDataIn                  - struct: most general metadata struct
% preprocCfg                  - struct: preprocessing related configuration


%% Loads all time domain data

% loads all time domain data and pads NaNs in cases where packets were lost
timeDomainDataStructOut.rawData = preprocLoadTdData(timeDomainData, ...
    metaDataIn, preprocCfg);

% obtain the lags
lag = getINSLags(cfg.str_sub);
t_abs = t_abs - lag;

%% Perform epoch rejection

% split the data into 2s long consecutive chunks
[t_stn_wnan, t_abs_wnan, raw_data_stn_wnan, raw_data_stn_wnan_filt, ...
    idx_start, idx_valid_epoch, t_start] = preprocArtifactRejectFirstPass(...
    t_stn, t_abs, raw_data_stn, raw_data_stn_filt, fs, cfg);

% remove chunks that are too isolated
[t_stn_wnan_corr, raw_data_stn_wnan_corr, ...
    raw_data_stn_wnan_filt_corr, idx_valid_epoch_corr] = ...
    preprocRemoveChunks(t_stn_wnan, raw_data_stn_wnan, ...
    raw_data_stn_wnan_filt, ...
    idx_start, idx_valid_epoch, cfg);


%% append to output structure

% original data
output.time = t_stn;
output.time_abs = t_abs;
output.raw_data = raw_data;
output.raw_data_filt = raw_data_filt;

% data before correction for isolated chunks
output.t_stn_wnan = t_stn_wnan;
output.t_abs_wnan = t_abs_wnan;
output.raw_data_stn_wnan = raw_data_stn_wnan;
output.raw_data_stn_wnan_filt = raw_data_stn_wnan_filt;

output.idx_valid_epoch = idx_valid_epoch;
output.idx_start = idx_start;

% output after correction for isolated chunks
output.t_stn_wnan_corr = t_stn_wnan_corr;
output.raw_data_stn_wnan_corr = raw_data_stn_wnan_corr;
output.idx_valid_epoch_corr = idx_valid_epoch_corr;

output.t_start = t_start;
output.raw_data_stn_wnan_filt_corr = raw_data_stn_wnan_filt_corr;


end