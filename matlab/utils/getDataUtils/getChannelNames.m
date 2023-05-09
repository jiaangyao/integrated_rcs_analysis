function vecStrChan = getChannelNames(timeDomainSettings)
%% getDataUtils script - loading of all channel names
% Loads channel name for all recording channels in the time domain based
% off timeDomainSettings
% 
% INPUT:
% timeDomainSettings    - table: time domain setting table from processRCS


%% Sanity checks

% sanity check to make sure the same channels are streamed in this file
if size(timeDomainSettings, 1) > 1
    % double check the metadata is oriented the right way
    assert(varDimCheckValidEntryExist(timeDomainSettings))
    assert(varDimCheckTallArray2D(timeDomainSettings.TDsettings{1}), ...
        'Double check the metadata shape, TDsettings should be a tall array')
    
    % check for equality across all entries
    vecStrChanTest = {timeDomainSettings.TDsettings{1}(:).chanOut};
    for i = 2:size(timeDomainSettings, 1)
        assert(isequal(vecStrChanTest, ...
            {timeDomainSettings.TDsettings{i}(:).chanOut}))
    end
else
    vecStrChanTest = {timeDomainSettings.TDsettings{1}(:).chanOut};
end

% check that there are four channels
assert(numel(vecStrChanTest) <= 4, 'There should be at most 4 channels')


%% Obtain the correct channel names

% obtain the string of all channels
vecStrChan = varDimConvertWideArraytoTall(vecStrChanTest);


end