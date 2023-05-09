function [subCortical, cortical] = strParExtractChanInfo(vecStrChan)
%% strParUtils script - extracts information about channels
% Parses the channel information and decide if they are cortical vs
% subcortical and referencing schemes based on the input regular expression
% pattern
% 
% INPUT:
% vecStrChan            - cell: cell of string of channel information
% regExpPattern         - string: string pattern for the regular expression


%% Parse the input cell array and extract the logical masks

% loop through all channels and parse separetely for cortical and
% subcortical patterns
subCortical.boolLocChan = logical([]);
subCortical.boolLocSandwitch = logical([]);
subCortical.boolCase = logical([]);

cortical.boolLocChan = logical([]);
cortical.boolLocSandwitch  = logical([]);
for i = 1:numel(vecStrChan)
    % parse which channels are subcortical
    [tokens, ~] = regexp(vecStrChan{i},  '+(\w*)-(\w*)', 'tokens', 'match');
    
    % first extract the string and check if case is contained in
    % referencing
    strChan1 = tokens{1}{1};
    strChan2 = tokens{1}{2};
    
    % parse channel 1 information
    if strcmpi(strChan1, 'C')
        boolCaseChan1 = true;
        idxChan1 = 'C';
    else
        boolCaseChan1 =  false;
        idxChan1 = str2double(strChan1);
    end

    % parse channel 2 information
    if strcmpi(strChan2, 'C')
        boolCaseChan2 = true;
        idxChan2 = 'C';
    else
        boolCaseChan2 =  false;
        idxChan2 = str2double(strChan2);
    end
    subCortical.boolCase = [subCortical.boolCase; ...
        boolCaseChan1 || boolCaseChan2];
        
    % sanity check
    if isnan(idxChan1) || isnan(idxChan2)
        error('Channel parsing fails')
    end

    % parse contical and subcortical
    if (idxChan1 <= 4 || boolCaseChan1) && (idxChan2 <= 4 || boolCaseChan2)
        % if current channel is subcortical
        % get the subcortical information first
        subCortical.boolLocChan = [subCortical.boolLocChan; true];
        if boolCaseChan1 || boolCaseChan2
            % if reference to case
            subCortical.boolLocSandwitch = ...
                [subCortical.boolLocSandwitch; false];
        elseif abs(idxChan1 - idxChan2) == 2
            % if bipolar & sandwitch
            subCortical.boolLocSandwitch = ...
                [subCortical.boolLocSandwitch; true];
        else
            % if bipolar & not sandwitch
            subCortical.boolLocSandwitch = ...
                [subCortical.boolLocSandwitch; false];
        end
        
        % now get the cortical channel information
        cortical.boolLocChan = [cortical.boolLocChan; false];
        cortical.boolLocSandwitch = [cortical.boolLocSandwitch; false];

    elseif idxChan1 >= 8 && idxChan2 >= 8
        % if current channel is cortical
        % get the subcortical information first
        subCortical.boolLocChan = [subCortical.boolLocChan; false];
        subCortical.boolLocSandwitch = [subCortical.boolLocSandwitch; false];

        % now get the cortical information
        cortical.boolLocChan = [cortical.boolLocChan; true];
        if abs(idxChan1 - idxChan2) == 2
            cortical.boolLocSandwitch = ...
                [cortical.boolLocSandwitch; true];
        else
            cortical.boolLocSandwitch = ...
                [cortical.boolLocSandwitch; false];
        end
    end
end


end