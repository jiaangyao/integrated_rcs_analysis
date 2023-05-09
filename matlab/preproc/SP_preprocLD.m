function output = SP_preprocLD(adaptiveData, output, cfg)


%% establish indexing

% obtain the actual sample by sample index of the adaptive data (because 
% of update rate)
idxLDChangeFull = [];
updateRateN = round(1e3 * currDetectorSetting.(strLDFieldName).updateRate / ...
    metaDataIn.powerMetaData.fftConfig.interval);
idxLDChange = find(abs(diff(adaptiveData.(sprintf('%s_output', strLDFieldName)))) > 0);
idxLDSkipped = find(diff(idxLDChange) > updateRateN);
for i = 1:numel(idxLDSkipped)
    % obtain current start and end indices
    if i == 1
        idxStart = 1;
    else
        idxStart = idxLDSkipped(i - 1) + 1;
    end
    idxEnd = idxLDSkipped(i);
    
    % append idx with all indices before
    idxLDChangeFull = [idxLDChangeFull; idxLDChange(idxStart:idxEnd)];

    % now add the missing entry (defaults to same as update rate)
    idxLDChangeFull = [idxLDChangeFull; updateRateN + idxLDChangeFull(end)];
end

% have to pad last discontinuity to end
idxLDChangeFull = [idxLDChangeFull; idxLDChange((idxLDSkipped(end) + 1):end)];
assert(all(diff(idxLDChangeFull) <= updateRateN), 'LD indexing failed');

% obtain the actual index with repeats
idxFull = repelem(1:(numel(idxLDChangeFull) + 1), ...
    [idxLDChangeFull(1); diff(idxLDChangeFull); size(adaptiveData, 1) - idxLDChangeFull(end)])';
assert(size(idxFull, 1) == size(adaptiveData, 1), 'LD indexing failed');
for i = 1:max(idxFull)
    idxCurr = idxFull == i;
    dataCurr = adaptiveData.(sprintf('%s_output', strLDFieldName))(idxCurr);
    assert(all(dataCurr == dataCurr(1)));
end


strCurrSide = output.metaData.genMetaData.strSide;

% run a check on whether the adaptive data exists
% obtain the time stamps for the adaptive data
n_sample_adap_abs = table2array(adaptiveData(:, 'newDerivedTime'));
t_adap = 1e-3 * (n_sample_adap_abs - n_sample_adap_abs(1));
t_adap_abs = table2array(adaptiveData(:, 'localTime'));

% obtain the actual adaptive data
vec_currrent_in = table2array(adaptiveData(:, 'CurrentProgramAmplitudesInMilliamps'));
vec_str_state = table2array(adaptiveData(:, 'CurrentAdaptiveState'));
ld0_data = table2array(adaptiveData(:, 'Ld0_featureInputs'));
ld1_data = table2array(adaptiveData(:, 'Ld1_featureInputs'));

% convert the vector of states in string to integers
vec_idx_state = ones(size(vec_str_state)) * NaN;
for i = 1:numel(vec_str_state)
    if strcmp(vec_str_state{i}, 'No State')
        continue
    elseif strcmp(vec_str_state{i}, 'State 0')
        vec_idx_state(i) = 0;
    elseif strcmp(vec_str_state{i}, 'State 1')
        vec_idx_state(i) = 1;
    elseif strcmp(vec_str_state{i}, 'State 2')
        vec_idx_state(i) = 2;
    elseif strcmp(vec_str_state{i}, 'State 3')
        vec_idx_state(i) = 3;
    elseif strcmp(vec_str_state{i}, 'State 4')
        vec_idx_state(i) = 4;
    elseif strcmp(vec_str_state{i}, 'State 5')
        vec_idx_state(i) = 5;
    elseif strcmp(vec_str_state{i}, 'State 6')
        vec_idx_state(i) = 6;
    elseif strcmp(vec_str_state{i}, 'State 7')
        vec_idx_state(i) = 7;
    elseif strcmp(vec_str_state{i}, 'State 8')
        vec_idx_state(i) = 8;
    else
        error("Unknown state")
    end
end

% now perform artifact rejection on the power data
[t_adap_wnan, t_adap_abs_wnan, ld0_data_wnan] = ...
    preprocArtifactRejectwIdx(t_adap, t_adap_abs, ld0_data, ...
    output.t_start, output.idx_valid_epoch_corr);


% also form state variable with collapsed states


%% if combo threshold then extract additional parameters

if output.metaData.adaptiveMetaData.LD0.(strCurrSide).boolCombo || ...
        output.metaData.adaptiveMetaData.LD1.(strCurrSide).boolCombo

    cfg.thresCombo = true;
    warning('Double check the weight vectors, note these are rounded down to nearest integer')
    
    % not supporting both combo
    if output.metaData.adaptiveMetaData.(strCurrSide).LD0.boolCombo && ...
            output.metaData.adaptiveMetaData.(strCurrSide).LD1.boolCombo
        error("NotImplementedError")
    end
    
    if output.adaptiveMetaData.(currStrSide).LD0.boolCombo
        weightVector = [6, 14];
        error("Double check the weight vector logic")
        LDCombo = table2array(adaptiveData(:, 'Ld0_output'));

    elseif output.adaptiveMetaData.(currStrSide).LD1.boolCombo
        weightVector = [6, 14];
        error("Double check the weight vector logic")
        LDCombo = table2array(adaptiveData(:, 'Ld1_output'));
    end

    output.weightVector = weightVector;
    output.LDCombo = LDCombo;
end


%% append to output

% now form the output structure;


% adaptive output before and after artifact rejection for STN bands
output.t_adap = t_adap;
output.t_adap_abs = t_adap_abs;
output.vec_currrent_in = vec_currrent_in;
output.vec_idx_state = vec_idx_state;

output.ld0_data = ld0_data;
output.ld1_data = ld1_data;

output.t_adap_wnan = t_adap_wnan;
output.t_adap_abs_wnan = t_adap_abs_wnan;
output.ld0_data_wnan = ld0_data_wnan;


end