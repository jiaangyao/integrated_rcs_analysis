function ldMetaDataStructOut = cfgGetLDInfo(strLD, currAdaptiveSetting, ...
    currDetectorSetting, adaptiveMetaDataIn, metaDataIn, boolReverseNonActiveLD)
%% getDataUtils script - loading of LD information from adaptive setting
% Loads LD information based on adaptive setting provided
%
% INPUT:
% strLD                     - string: name of the current LD
%                               LD0|LD1
% currAdaptiveSetting       - table: last row of adaptive setting table 
%                               from processRCS
% currDetectorSetting       - table: last row of LD setting table 
%                               from processRCS
% adaptiveMetaDataIn        - struct: metadata containing relevant
%                               information for both LDs
% metaDataIn                - struct: most general metadata struct
% boolReverseNonActiveLD    - boolean: whether to reverse rule for
%                               LD -> high LD == low stim


%% Sanity check

% enforce that LD queried is LD0/LD1
if ~contains(strLD, {'LD0'; 'LD1'}, 'IgnoreCase', true)
    error('Can only parse information for LD0/LD1, current input: %s', strLD)
end


%% Parse LD information from various setting tables

% first see which LD is being queried to see whether to look down
% row/column
if strcmpi(strLD, 'LD0')
    strLDFieldName = 'Ld0';
    idxValidStimLDCurr = ...
        varDimConvertWideArraytoTall(adaptiveMetaDataIn.idxValidStim(1, :));
    fullStateNameValidLDCurr = ...
        varDimConvertWideArraytoTall(adaptiveMetaDataIn.fullStateNameValid(1, :));
    fullStimValidLDCurr = ...
        varDimConvertWideArraytoTall(adaptiveMetaDataIn.fullStimValid(1, :));
else
    strLDFieldName = 'Ld1';
    idxValidStimLDCurr = adaptiveMetaDataIn.idxValidStim(:, 1);
    fullStateNameValidLDCurr = adaptiveMetaDataIn.fullStateNameValid(:, 1);
    fullStimValidLDCurr = adaptiveMetaDataIn.fullStimValid(:, 1);
end

% check for current LD enabled or no
if sum(idxValidStimLDCurr) >= 2
    ldMetaDataStructOut.boolEnabled = true;
else
    ldMetaDataStructOut.boolEnabled = false;
end

% check if current LD is dual threshold or no
if all(idxValidStimLDCurr)
    ldMetaDataStructOut.boolDualThresh = true;
else
    ldMetaDataStructOut.boolDualThresh = false;
end

% check if current LD is the active LD or no
if numel(unique(fullStimValidLDCurr)) == numel(fullStimValidLDCurr) && ...
        numel(unique(adaptiveMetaDataIn.fullStimValid)) ~= 1
    ldMetaDataStructOut.boolActive = true;
else
    ldMetaDataStructOut.boolActive = false;
end

% obtain detector input as bool array
boolPBisLDCurrInput = [];
detectionInputs = flip(currDetectorSetting.(strLDFieldName).detectionInputs_BinaryCode);
for i = 1:numel(detectionInputs)
    boolCurrPBActive = strcmp(detectionInputs(i), '1');
    boolPBisLDCurrInput = [boolPBisLDCurrInput, boolCurrPBActive];
end
boolPBisLDCurrInput = logical(boolPBisLDCurrInput);

% check if LD0 is combo or no
if sum(boolPBisLDCurrInput) > 1
    ldMetaDataStructOut.boolCombo = true;
    ldMetaDataStructOut.strLD = 'LD Combo';
    for i = 1:sum(boolPBisLDCurrInput)
        ldMetaDataStructOut.strLDFull{i} = sprintf('LD%d', i-1);
    end

else
    ldMetaDataStructOut.boolCombo = false;
    ldMetaDataStructOut.strLD = strLD;
    ldMetaDataStructOut.strLDFull = {ldMetaDataStructOut.strLD};
end

% obtain current input power band
fullInputIdx = 1:numel(boolPBisLDCurrInput);
fullInputIdx = fullInputIdx(boolPBisLDCurrInput);
for i = 1:numel(fullInputIdx)
    ldMetaDataStructOut.strCh{i} = ...
        metaDataIn.powerMetaData.strVecChanwPB{fullInputIdx(i)};
    ldMetaDataStructOut.strPB{i} = ...
        metaDataIn.powerMetaData.powerBandsInHz{fullInputIdx(i)};
end

% check if current LD is subcortical
idxCurrChan = strcmpi(ldMetaDataStructOut.strCh, ...
    metaDataIn.timeDomainMetaData.vecStrChan);
if metaDataIn.timeDomainMetaData.subCortical.boolLocChan(idxCurrChan)
    ldMetaDataStructOut.boolSubCor = true;
else
    ldMetaDataStructOut.boolSubCor = false;
end

% check if current LD is reverse or no (i.e. gamma-based LD that's reversed)
% first check if the stim amplitude are reversed
[~, idxSortStimAmp_LDCurr] = sort(fullStimValidLDCurr);
idxRev = varDimConvertWideArraytoTall(numel(fullStimValidLDCurr):-1:1);
boolActiveStimReverse = all(idxSortStimAmp_LDCurr == idxRev);
if strcmpi(strLD, 'LD0') && boolActiveStimReverse
    ldMetaDataStructOut.boolReverse = true;
elseif strcmpi(strLD, 'LD1') && boolReverseNonActiveLD
    ldMetaDataStructOut.boolReverse = true;
else
    ldMetaDataStructOut.boolReverse = false;
end

% check if current LD is fake adaptive or no
if numel(unique(adaptiveMetaDataIn.fullStimValid)) == 1
    ldMetaDataStructOut.boolFakeAdaptive = true;
    ldMetaDataStructOut.stimLevelCorr = ...
        getFakeAdaptiveAmp(metaDataIn.genMetaData.strSubject, ...
        metaDataIn.genMetaData.strStep, metaDataIn.genMetaData.strRound);
    t1 = 1;
end

% obtain other numerical settings
ldMetaDataStructOut.thresh = ...
    currDetectorSetting.(strLDFieldName).biasTerm;
ldMetaDataStructOut.updateRate = ...
    currDetectorSetting.(strLDFieldName).updateRate;
ldMetaDataStructOut.onsetDuration = ...
    currDetectorSetting.(strLDFieldName).onsetDuration;
ldMetaDataStructOut.terminationDuration = ...
    currDetectorSetting.(strLDFieldName).terminationDuration;

% also append stim-related information
ldMetaDataStructOut.stimLevel = ...
    sort(unique(adaptiveMetaDataIn.fullStimValid)); 
ldMetaDataStructOut.stimRate = currAdaptiveSetting.stimRate;
ldMetaDataStructOut.rampRate = [currAdaptiveSetting.deltas{1}(1).rise, ...
    currAdaptiveSetting.deltas{1}(1).fall];
ldMetaDataStructOut.statesFull = fullStateNameValidLDCurr;
ldMetaDataStructOut.idxShift = metaDataIn.powerMetaData.idxShift;


end