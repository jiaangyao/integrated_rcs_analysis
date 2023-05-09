function [relTimewNAN_CORR, rawDatawNAN_CORR, rawDataFiltwNAN_CORR, idxValidEpoch_CORR] = ...
    preprocRemoveChunks(relTimewNAN, rawDatawNAN, rawDataFiltwNAN, ...
    idxStart, idxValidEpoch, cfg)

% sanity check by computing chunks that are too isolated
relTimewNAN_CORR = relTimewNAN;
relTimeFiltwNAN_CORR = relTimewNAN;

rawDatawNAN_CORR = rawDatawNAN;
rawDataFiltwNAN_CORR = rawDataFiltwNAN;

idxValidEpoch_CORR = idxValidEpoch;
idxValidEpoch_FILT_CORR = idxValidEpoch;

% loop through all patterns for cleaning
for i = 1:numel(cfg.vec_epoch_pattern)
    epochPattCurr = cfg.vec_epoch_pattern{i};

    [relTimewNAN_CORR, rawDatawNAN_CORR, idxValidEpoch_CORR] = ...
        correctIsolatdChunks(relTimewNAN_CORR, rawDatawNAN_CORR, ...
        idxStart, idxValidEpoch_CORR, epochPattCurr);

    [~, rawDataFiltwNAN_CORR, ~] = ...
        correctIsolatdChunks(relTimeFiltwNAN_CORR, rawDataFiltwNAN_CORR, ...
        idxStart, idxValidEpoch_FILT_CORR, epochPattCurr);
end


end