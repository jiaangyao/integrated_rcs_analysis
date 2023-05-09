function strTime = strParRegExpMarker(strMarker, strRegExp, idxParsewoSpace)
%% strParUtils script - extracts time information from marker text
% Parses the patient provided marker information
% 
% INPUT:
% strMarker            - string: patient provided event marker text
% strRegExp            - string: regular expression text to be parsed
% idxParsewoSpace      - double: starting index for parsing without space

%% Perform regular expression parsing

% parse and get rid of extra space if exists
strTime = regexp(strMarker, strRegExp, 'match');
if contains(strTime{1}, ' ')
    strTime = strTime{1}((idxParsewoSpace+1):end);
else
    strTime = strTime{1}(idxParsewoSpace:end);
end

end