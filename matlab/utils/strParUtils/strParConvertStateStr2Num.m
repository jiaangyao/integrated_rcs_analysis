function vecIdxState = strParConvertStateStr2Num(vecStrState)
%% strParUtils script - converts state information from string to num
% Loads Device ID based on subject name and which side the recording is on
% Note: all of the Device IDs in this script are hardcoded so make sure to
% very them
% 
% INPUT:
% vecStrState           - cell: cell of string of state information


%% Convert state from string to numerical values

% for loop through all entries
vecIdxState = ones(size(vecStrState)) * NaN;
for i = 1:numel(vecStrState)
    if strcmp(vecStrState{i}, 'No State')
        continue
    elseif strcmp(vecStrState{i}, 'State 0')
        vecIdxState(i) = 0;
    elseif strcmp(vecStrState{i}, 'State 1')
        vecIdxState(i) = 1;
    elseif strcmp(vecStrState{i}, 'State 2')
        vecIdxState(i) = 2;
    elseif strcmp(vecStrState{i}, 'State 3')
        vecIdxState(i) = 3;
    elseif strcmp(vecStrState{i}, 'State 4')
        vecIdxState(i) = 4;
    elseif strcmp(vecStrState{i}, 'State 5')
        vecIdxState(i) = 5;
    elseif strcmp(vecStrState{i}, 'State 6')
        vecIdxState(i) = 6;
    elseif strcmp(vecStrState{i}, 'State 7')
        vecIdxState(i) = 7;
    elseif strcmp(vecStrState{i}, 'State 8')
        vecIdxState(i) = 8;
    else
        error("Unknown state encountered: %s", vecStrState{i})
    end
end

end