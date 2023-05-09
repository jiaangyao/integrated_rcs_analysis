function markerParseSettings = cfgGetMarkerParseSettings(metaDataIn)
%% cfgUtils script - obtain marker parsing settings for RCS patients
% Obtain marker parsing settings and configs (hard-coded in advance) for
% RCS patients
% 
% INPUT:
% metaDataIn            - struct: most general metadata struct


%% obtain marker parsing settings

% if adaptive data is enabled
if metaDataStructOut.adaptiveMetaData
    markerParseSettings.isEnabled = true;
    if strcmp(metaDataIn.genMetaData.strSubject, 'RCS02')
        % parse dyskinesia related markers for RCS02
        markerParseSettings.vecStrKeyContain = {'dsk', 'dys', 'dyk', 'dks'};
        markerParseSettings.boolAddStop = true;
        markerParseSettings.defaultDuration = 15;

    elseif strcmp(metaDataIn.genMetaData.strSubject, 'RCS14')
        % parse tremor related markers for RCS14;
        markerParseSettings.vecStrKeyContain = {'trem'};
        markerParseSettings.boolAddStop = false;
        markerParseSettings.defaultDuration = 0;

    else
        error('Undefined subject: %s', metaDataIn.genMetaData.strSubject)
    end

% if no adaptive data is present
else
    % create empty structure
    markerParseSettings.isEnabled = false;
end


end