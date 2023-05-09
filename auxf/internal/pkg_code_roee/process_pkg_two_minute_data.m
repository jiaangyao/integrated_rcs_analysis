function process_pkg_two_minute_data()
%% how data are categorized to different states using PKG 
% pkgOffMeds  = bksabs > 32 & bksabs < 80;
% pkgOnMeds   = bksabs <= 20 & bksabs > 0;
%{
    We use BKS>26<40 (BKS=26 =UPDRS III~30)as a marker of
    ?OFF? and >32<40 as (BKS=32 =UPDRS III~45)marker of very OFF
    We use DKS>7 as a marker of dyskinesia and > 16 as significant dyskinesia
    Generally when B70lyoyyKS>26, DKS will be low.
    We don?t usually use the terminology of OFF/On/dyskinesia use in diaries
    because they are categorical states compared to a continuous variable.
    If I can ask you the same question for UPDRS and AIMS score
    what cut-off would you like to use to indicate those
    same states and then I can give you approximate numbers for the BKS DKS.
    We have good evidence thatTreatable bradykinesia
    (i.e. presumable OFF according to a clinician) is when the
     BKS>26 (or <-26 as per the csv files)
    Good control (i.e. neither OFF nor ON) is when BKS <26 AND DKS<7
    Dyskinesia is when DKS>7 and BKS <26.
    However you should not use single epochs alone.
    We tend to use the 4/7 or 3/5 rule ?
    that is use take the first 7 epochs of BKS (or DKS),
    then the middle epoch will be ?OFF? if 4/7 of the epochs >26.
    Slide along one and apply the rule again etc.
    Mal Horne
    malcolm.horne@florey.edu.au
    Wed 7/24/2019 7:12 PM email
    subjet: More data for UCSF

    DKS ? and note the 75th percentile of controls" - what is the number for this?
    RE BKS ? see bands of activity ? suggest you make more correlation for BKS in range 18-40
    As said before, above 80 = sleep and between 40-80 = inactivity (i.e. on couch)
    Tue 7/9/2019 5:10 PM
    subject:
    sessions with detailed data
%}

%% pre amble 
clc;
close all;

%% load the data
[rootdir, ~] = fileparts(pwd);
masterTable = readtable(fullfile(rootdir,'data','pkg_master_list.csv'));

%% extract PKG 2 minute score data and save as .mat file
re_extract_db = 1; % change to 1 to re-extract the database
if re_extract_db
    pkgDB = table();
    filenames = masterTable.FileName;
    idxkeep = ~cellfun(@(x) isempty(x), filenames);
    masterTable = masterTable(idxkeep,:);
    filenames = masterTable.FileName;
    cntpt = 1;
    for f = 1:length(filenames)
        newstr = strrep(filenames{f},'''',''); % extract just the code name
        strfind = newstr(end-5:end);
        % find scores
        ff_scores = findFilesBVQX(rootdir, ['scores*' strfind '*.csv']);
        ff_doses = findFilesBVQX(rootdir, ['dose*' strfind '*.csv']);
        if ~isempty(ff_scores)
            % read pkg
            pkgTable = readtable(ff_scores{1});
            doseTable = readtable(ff_doses{1});
            timesPKG = pkgTable.Date_Time;
            pkgTable.reportID = repmat({strfind},size(pkgTable,1),1);
            timesPKG.TimeZone = 'America/Los_Angeles';
            
            % get rid of NaN data (it's empty on startup
            pkgTable = pkgTable(~isnan(pkgTable.BK),:);
            
            % get rid of off wrist data
            pkgTable = pkgTable(~pkgTable.Off_Wrist,:);
            
            if size(pkgTable,1) > 40 % min size 40  values = 80 minutes
                
                % use 3/5 rule to look at 10 minute epochs
                for i = 3:(size(pkgTable,1)-2)
                    bkvals = pkgTable.BK(i-2:i+2);
                    dkvals = pkgTable.DK(i-2:i+2);
                    tremor = pkgTable.Tremor(i-2:i+2);
                    % off - bks under 26 and over 80 (sleep)
                    cnt = 1;
                    state = {};
                    if sum(bkvals <= -26 & bkvals >= -80) >=3 % off
                        state{cnt} = 'off'; cnt = cnt +1;
                    end
                    %     Good control (i.e. neither OFF nor ON) is when BKS <26 AND DKS<7
                    if sum(bkvals >= -26 & dkvals <= 16) >=3 % on #### juan: why DBVALS <= 16 instead of 7
                        state{cnt} = 'on'; cnt = cnt +1;
                    end
                    % dyskinesia mild
                    if sum(bkvals >= -26 & (dkvals >= 7 & dkvals < 16)) >=3 % on
                        state{cnt} = 'dyskinesia mild'; cnt = cnt +1;
                    end
                    % dyskinesia severe
                    if sum(bkvals >= -26 & dkvals >= 16) >=3 % on
                        state{cnt} = 'dyskinesia severe'; cnt = cnt +1;
                    end
                    %    tremor
                    if sum(tremor) >=3 % tremor
                        state{cnt} = 'tremor'; cnt = cnt +1;
                    end
                    %   sleep
                    if  sum(bkvals < -80) >=3 % off
                        state{cnt} = 'sleep'; cnt = cnt +1;
                    end
                    tremorScore = mean(tremor);
                    if length(state)==2
                        x = 2;
                    end
                    stateLens(i) = length(state);
                    if isempty(state)
                        states{i,1} = 'uncategorized';
                    else
                        stateout = '';
                        for s = 1:length(state)
                            if s == 1
                                stateout = [stateout state{s}];
                            else
                                stateout = [stateout ' ' state{s}];
                            end
                        end
                        states{i,1} = stateout;
                    end
                end
                
                % save table
                states = states(3:end);
                idxsave = 3:(size(pkgTable,1)-2);
                pkgTable = pkgTable(idxsave,:);
                pkgTable.states = states;
                clear states;
                times = [pkgTable.Date_Time(1) pkgTable.Date_Time(end)];
                times.Format = 'uuuu-MM-dd';
                savedir = fullfile(rootdir,'results','processed_data');
                if iscell(masterTable.initialProgrammingDate)
                    initial_programming_date = datetime(masterTable.initialProgrammingDate{f},'Format','MM/dd/yy');
                else
                    initial_programming_date = datetime(masterTable.initialProgrammingDate(f),'Format','MM/dd/yy');
                end
                if iscell(masterTable.surgeryDate)
                    surger_date = datetime(masterTable.surgeryDate{f},'Format','MM/dd/yy');
                else
                    surger_date = datetime(masterTable.surgeryDate(f),'Format','MM/dd/yy');
                end
                if times(1) < surger_date
                    struse = 'before_implant';
                elseif initial_programming_date > times(1)
                    struse = 'before-programming';
                elseif initial_programming_date < times(1)
                    struse = 'after-programming';
                end
                stateSvName = sprintf('%s_%s-hand_%s__%s__%s.mat',masterTable.Patient{f},masterTable.Side{f},...
                    struse,times(1),times(2));
                savefn = fullfile(savedir,stateSvName);
                metaData = masterTable(f,:);
                save(savefn,'pkgTable','metaData','doseTable');
                
                pkgDB.patient{cntpt} = masterTable.Patient{f};
                pkgDB.side{cntpt} = masterTable.Side{f};
                pkgDB.timerange(cntpt,:)   = times;
                pkgDB.surgery_date(cntpt)   = surger_date;
                pkgDB.initial_programming_date(cntpt)   = initial_programming_date;
                pkgDB.date_details{cntpt} = struse;
                pkgDB.notes{cntpt} = masterTable.Notes{f};
                pkgDB.pkg_identifier{cntpt} = strfind;
                pkgDB.savefn{cntpt}      = stateSvName;
                cntpt = cntpt + 1;
            end
        end
    end
    savedir = fullfile(rootdir,'results','processed_data');
    save(fullfile(savedir,'pkgDataBaseProcessed.mat'),'pkgDB');
    writetable(pkgDB,fullfile(savedir,'pkgDataBaseProcessed.csv'));
end


end