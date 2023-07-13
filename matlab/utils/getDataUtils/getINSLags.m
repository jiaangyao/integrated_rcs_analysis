function lag = getINSLags(str_subject)

if strcmp(str_subject, 'RCS02')
    lag = seconds(0 * 3600 + 22 * 60 + 37);

elseif strcmp(str_subject, 'RCS17')
    lag = seconds(0 * 3600 + 4 * 60 + 36);

else
    lag = seconds(0);
    warning("Need to check for INS lag for subject %s", str_subject)
end

end