function cutoffFreq = getStimFiltCutoff(stimFreq, strLoc)
%% getDataUtils script - loading of filter cutoff frequency based on stim freq
% Loads LPF cutoff frequency based on stimulation frequency
% 
% INPUT:
% stimFreq      - double: stim frequency in Hz
% strLoc        - string: location of the filters
%                   'subCortical'|'cortical'


%% warning and sanity check
% warning('Note: Device IDs are hardcoded right now and make sure to verify them')
if ~(strcmpi(strLoc, 'subCortical') || strcmpi(strLoc, 'cortical'))
    error('Location can only be subCortical or cortical, current input: %s', strLoc)
end


%% actual comparison and querying for device ID
if strcmpi(strLoc, 'subCortical')
    if stimFreq > 128 && stimFreq < 140
        % for stim freq in the 130s range
        cutoffFreq = 100;
    elseif stimFreq > 180 && stimFreq < 198
        cutoffFreq = 110;
    else
        error('Unencountered stim freq: %d', stimFreq);
    end

elseif strcmpi(strLoc, 'cortical')
    if stimFreq > 128 && stimFreq < 140
        % for stim freq in the 130s range
        cutoffFreq = 110;
    elseif stimFreq > 180 && stimFreq < 198
        cutoffFreq = 120;
    else
        error('Unencountered stim freq: %d', stimFreq);
    end

end


end