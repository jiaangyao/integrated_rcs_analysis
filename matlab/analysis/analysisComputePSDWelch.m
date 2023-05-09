function analysisComputePSDWelch(outputStructCurr, varargin)
%% Input parsing
% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'boolUseFilt', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'boolUseReref', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));
addParameter(p, 'strRerefMethod', 'CAR', ...
    @(x) validateattributes(x, {'char'}, {'nonempty'}));

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
boolUseFilt = p.Results.boolUseFilt;  
boolUseReref = p.Results.boolUseReref;
strRerefMethod = p.Results.strRerefMethod;

%%

% obtain the metadata
fs = outputStructCurr.fs;                                                   % sampling rate
strChan = outputStructCurr.vec_str_chan;                                    % name of the channel

% obtain the raw data or the filtered data depending on user flag
if boolUseFilt
    rawDataCurr = outputStructCurr.rawDataFilt;

else
    rawDataCurr = outputStructCurr.rawData;
end

% optionally apply re-referencing depending on user flag
if boolUseReref
    if strcmpi(strRerefMethod, 'CAR')
        rawDataCurr = rawDataCurr - mean(rawDataCurr, )
    elseif

else

end


end