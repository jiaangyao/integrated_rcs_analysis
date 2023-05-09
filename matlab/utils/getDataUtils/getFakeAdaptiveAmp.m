function stimLevelCorr = getFakeAdaptiveAmp(strSubject, strSide, ...
    strStep, strRound)
%% getDataUtils script - loading of stimluation amplitude for fake adaptive days
% Obtain hardcoded stimulation amplitude based on user definition
% 
% INPUT:
% strSubject            - string: string for the name of the subject, e.g. RCS01


%% Provide hardcoded fake adaptive low and high stim values

if strcmpi(strSubject, 'RCS02')
    if strcmpi(strSide, 'L')
    error("NotImplementedError");

elseif strcmpi(strSubject, 'RCS14')
    % For RCS14, Step6
    if contains(strStep, 'Step6', 'IgnoreCase', true)
        % for round 4, 5, 6
        if any(strcmpi(strRound, {'Round4', 'Round5', 'Round6'}))
            warning('Hardcoded stim level for fake adaptive data')
            stimLevelCorr = [3.6, 3.9];
        else
            error("NotImplementedError");
        end
    else
        error("NotImplementedError");
    end
else
    error("NotImplementedError");
end


end
