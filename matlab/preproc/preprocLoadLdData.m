function output = preprocLoadLdData(output, AdaptiveData, timeStart, ...
    idxValidEpoch_CORR, cfg)

% run a check on whether the adaptive data exists
if size(AdaptiveData, 1) ~= 0
    bool_adap_valid = true;

    % obtain the time stamps for the adaptive data
    n_sample_adap_abs = table2array(AdaptiveData(:, 'newDerivedTime'));
    t_adap = 1e-3 * (n_sample_adap_abs - n_sample_adap_abs(1));
    t_adap_abs = table2array(AdaptiveData(:, 'localTime'));

    % obtain the actual adaptive data
    vec_currrent_in = table2array(AdaptiveData(:, 'CurrentProgramAmplitudesInMilliamps'));
    vec_str_state = table2array(AdaptiveData(:, 'CurrentAdaptiveState'));
    ld0_data = table2array(AdaptiveData(:, 'Ld0_featureInputs'));
    ld1_data = table2array(AdaptiveData(:, 'Ld1_featureInputs'));

    ld0_thres_low = table2array(AdaptiveData(:, 'Ld0_lowThreshold'));
    ld0_thres_high = table2array(AdaptiveData(:, 'Ld0_highThreshold'));
    ld1_thres_low = table2array(AdaptiveData(:, 'Ld1_lowThreshold'));
    ld1_thres_high = table2array(AdaptiveData(:, 'Ld1_highThreshold'));

    % convert the vector of states in string to integers
    vec_idx_state = loadStateRCS(vec_str_state);

    % now perform artifact rejection on the adaptive data
    [t_adap_wnan, t_adap_abs_wnan, ld0_data_wnan] = ...
        preprocArtifactRejectwIdx(t_adap, t_adap_abs, ld0_data, ...
        timeStart, idxValidEpoch_CORR);

% otherwise skip the whole process
else

    % in fact try to estimate all parameters based on the power band data
    bool_adap_valid = false;

    t_adap = t_pow;
    t_adap_abs = t_pow_abs;
    
    % re-estimate LD0 from the STN power bands
    fs_pow_curr = 1/mode(diff(t_pow));
    ld0_data = ones(size(pow_data_stn)) * NaN;

    % now try to estimate LD1 from the motor power bands
    ld1_data = ones(size(pow_data_motor)) * NaN;

    % TODO: note that these are hardcoded
    ld0_thres_low = NaN;
    ld0_thres_high = NaN;
    ld1_thres_low = NaN;
    ld1_thres_high = NaN;

    t_adap_wnan = t_pow_wnan;
    t_adap_abs_wnan = t_pow_abs_wnan;
    ld0_data_wnan = ones(size(pow_data_stn)) * NaN;

    % estimate the states based on the thresholds
    vec_currrent_in = ones(size(t_adap)) * NaN;
    vec_idx_state = ones(size(t_adap)) * NaN;

end

end