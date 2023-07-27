function output = SP_preprocPB(PowerData, output, cfg)
%% Load and process power data

% obtain the time stamps for the power data
n_sample_pow_abs = table2array(PowerData(:, 'newDerivedTime'));
t_pow = 1e-3 * (n_sample_pow_abs - n_sample_pow_abs(1));
t_pow_abs = table2array(PowerData(:, 'localTime'));


% obtain the lags
lag = getINSLags(cfg.str_sub);
t_pow_abs = t_pow_abs - lag;

% obtain the actual power data
pow_data_all = table2array(PowerData(:, contains(PowerData.Properties.VariableNames, 'Band')));
pow_data_stn = pow_data_all(:, 1:4);
pow_data_motor = pow_data_all(:, 5:8);

% now perform artifact rejection on the power data
[t_pow_wnan, t_pow_abs_wnan, pow_data_stn_wnan] = ...
    preprocArtifactRejectwIdx(t_pow, t_pow_abs, pow_data_stn, ...
    output.t_start, output.idx_valid_epoch_corr);

%% now append to output structure

% power output before and after artifact rejection for STN bands
output.t_pow = t_pow;
output.t_pow_abs = t_pow_abs;
output.pow_data_stn = pow_data_stn;
output.pow_data_motor = pow_data_motor;

output.t_pow_wnan = t_pow_wnan;
output.t_pow_abs_wnan = t_pow_abs_wnan;
output.pow_data_stn_wnan = pow_data_stn_wnan;

end