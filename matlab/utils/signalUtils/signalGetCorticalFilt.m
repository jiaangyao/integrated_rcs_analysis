function vecFiltOut = signalGetCorticalFilt(metaDataIn, preprocCfg)
%% signalUtils script - obtain filters corresponding to cortical data
% Obtain filters corresponding to cortical data based on user-provided 
% flags
%
% INPUT:
% metaDataIn            - struct: most general metadata struct
% preprocCfg            - struct: preprocessing related configuration


%% obtain filters for cortical regions

% initalize empty cell for filters
vecFiltOut = {};

% calculate default filter order
defaultFiltOrder = 2^(nextpow2(metaDataIn.timeDomainMetaData.fs * ...
        preprocCfg.cortical.filtLen));

% first check if stim is enabled and if enabled include a lowpass
if metaDataIn.stimMetaData.isEnabled
    % if enabled then check for stim frequency and apply a LPF
    stimFreq = unique(metaDataIn.stimMetaData.stimLogSettings.stimFreq);
    if numel(stimFreq) == 0
        error('Should be more than one stim frequency')
    elseif numel(stimFreq) > 1
        stimFreq = mode(metaDataIn.stimMetaData.stimLogSettings.stimFreq);
    end
    
    % next initialize filter
    filterLPFStim.strFiltName = 'lowpass';
    filterLPFStim.cutoffFreq = getStimFiltCutoff(stimFreq, 'cortical');
    filterLPFStim.filtOrder = defaultFiltOrder;
    
    % append to outer list
    vecFiltOut{end + 1} = filterLPFStim;
end

% next check whether user want to perform additional LPF
if preprocCfg.cortical.boolHPF
    assert(~isnan(preprocCfg.cortical.cutoffFreqHPF), 'Should be a number');
    
    % next initialize filter
    filterHPF.strFiltName = 'highpass';
    filterHPF.cutoffFreq = preprocCfg.cortical.cutoffFreqHPF;
    filterHPF.filtOrder = defaultFiltOrder;

    % append to outer list
    vecFiltOut{end + 1} = filterHPF;
end

% finally check whether user want to perform bandstop filtering
if preprocCfg.cortical.boolBSF
    assert(~any(isnan(preprocCfg.cortical.cutoffFreqBSF)), 'Should be a number');

    % next initialize filter
    filterBSF.strFiltName = 'bandstop';
    filterBSF.cutoffFreq = preprocCfg.cortical.cutoffFreqBSF;
    filterBSF.filtOrder = defaultFiltOrder;

    % append to outer list
    vecFiltOut{end + 1} = filterBSF;
end


end