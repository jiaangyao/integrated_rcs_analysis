function motordiary= GetMotorDiary(p, dt)

fprintf('\n> Loading Motor Diary Data');

if ~iscell(p)
    p = {p};
    dt = {dt};
end
for iP = 1:length(p)
    opts = detectImportOptions(p{iP});
    opts.VariableTypes{1}='datetime';
    [opts.VariableTypes{2:end}] = deal('double');
    
    md{iP} = readtable(p{iP}, opts);
    
    dtcell = strsplit(dt{iP}, '/');
    m = str2num(dtcell{1});
    d = str2num(dtcell{2});
    y = str2num(dtcell{3});
    
        md{iP}{:,1}.Month = 1;
    md{iP}{:,1}.Day = 1;
    md{iP}{:,1}.Year = 2020;
    
    md{iP}{:,1}.Month = m;
    md{iP}{:,1}.Day = d;
    md{iP}{:,1}.Year = y;
    
    for iC = 2:size(md{iP},2)
        boolNAN = isnan(md{iP}{:,iC});
        md{iP}{boolNAN,iC} = 0;
    end

    if iP == 1
        motordiary = md{iP};
    else
        try
        motordiary = [motordiary;md{iP}];
        catch
            error('Check and make sure there are the same columns for the two days of motor diaries');
        end
    end
end
motordiary = sortrows(motordiary, 1);
motordiary.Properties.VariableNames{1} = 'time';
fprintf('\n');
