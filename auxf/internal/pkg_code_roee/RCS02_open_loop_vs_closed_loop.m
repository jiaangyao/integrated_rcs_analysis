function RCS02_open_loop_vs_closed_loop()
%% load data 
[rootdir, ~] = fileparts(pwd);
savedir = fullfile(rootdir,'results','processed_data');
load(fullfile(savedir,'pkgDataBaseProcessed.mat'),'pkgDB');

uniqePatients = unique(pkgDB.patient);
uniqeSides    = unique(pkgDB.side);
close all;
clc;

%% % agregate all pkg data into one big table


pkgHugeTable = table();
for p  = 1:size(pkgDB,1)
    ff = findFilesBVQX( savedir, pkgDB.savefn{p}) ;
    load(ff{1});
    nrows = size(pkgTable,1);
    pkgTable.patient =  repmat(pkgDB.patient(p),nrows,1);
    pkgTable.date_details =  repmat(pkgDB.date_details(p),nrows,1);
    pkgTable.side =  repmat(pkgDB.side(p),nrows,1);
    pkgTable.timerange =  repmat(pkgDB.timerange(p,:),nrows,1);
    % add the dose information to the table
    for d = 1:size(doseTable)
        if ~isnan( doseTable.Dose(d))
            idxday = pkgTable.Day == doseTable.Day(d);
            if sum(idxday) > 1
                dayDates = pkgTable.Date_Time(idxday,:);
                yearUse = year(dayDates(1));
                monthUse = month(dayDates(1));
                dayUse = day(dayDates(1));
                [hourUse,minUse,secUse] = hms(doseTable.Reminder(d));
                dateReminder = datetime( sprintf('%d/%0.2d/%0.2d %0.2d:%0.2d:%0.2d',...
                    yearUse,monthUse,dayUse,hourUse,minUse,secUse),...
                    'Format','yyyy/MM/dd HH:mm:ss');
                
                [hourUse,minUse,secUse] = hms(doseTable.Dose(d));
                dateDose = datetime( sprintf('%d/%0.2d/%0.2d %0.2d:%0.2d:%0.2d',...
                    yearUse,monthUse,dayUse,hourUse,minUse,secUse),...
                    'Format','yyyy/MM/dd HH:mm:ss');
                [value,idx] = min(abs(dateDose-pkgTable.Date_Time));
                pkgTable.date_dose(idx) = dateDose;
                pkgTable.date_reminder(idx) = dateReminder;
            end
        end
    end
    if ~isfield(pkgTable,'date_dose') % in cased dose data doesn't exist for this subject
        pkgTable.date_dose(1) = NaT;
        pkgTable.date_reminder(1) = NaT;
    end
    
    if p == 1
        pkgHugeTable = pkgTable;
        clear pkgTable;
    else
        pkgHugeTable = [pkgHugeTable  ; pkgTable];
        clear pkgTable;
    end
end
% if you get a dyskinesia value that is 0 - and you log that
% you get -inf to fix this - change all dyskinesia values that are 0 to a 1
% so that you get a zero when you log it.
idxzero = pkgHugeTable.DK == 0;
pkgHugeTable.DK(idxzero) = 1;


sortedTbl = sortrows(pkgDB,{'patient','side','pkg_identifier','timerange'});


%% get PKG data and put in structure 
% relevant dates 
% 2019-04-24 - before implant - not clear which hand - maybe L (labeled NA)
% 2020-06-10 - adptive on right STN OL L stn, watch worn on L hand 
% 2020-05-26 - OL worn on L side on stim
% + get two adaptive runs bebfore porgramming on L 
idxPatient = strcmp(pkgHugeTable.patient,'RCS02');
pkgRC = pkgHugeTable(idxPatient,:);

pkgData = {};
cnt = 1; 

% baseline 
tableLabel{cnt} = 'off stim baseline'; 
idxUse = year(pkgRC.timerange(:,1)) == 2019 & ...
        month(pkgRC.timerange(:,1)) == 4 & ... 
        day(pkgRC.timerange(:,1)) == 24;
pkgData{cnt} = pkgRC(idxUse,:);
cnt = cnt + 1; 

% before stim  
tableLabel{cnt} = 'off stim post-op'; 
idxUse = strcmp(pkgRC.date_details,'before-programming') & ... 
    strcmp(pkgRC.side,'L');
pkgData{cnt} = pkgRC(idxUse,:);
cnt = cnt + 1; 

% on stim open loop  
tableLabel{cnt} = 'on stim open loop'; 
idxUse = year(pkgRC.timerange(:,1)) == 2020 & ...
        month(pkgRC.timerange(:,1)) == 5 & ... 
        day(pkgRC.timerange(:,1)) == 26 & ... 
        strcmp(pkgRC.side,'L');
pkgData{cnt} = pkgRC(idxUse,:);
cnt = cnt + 1; 

% on stim adaptive   
tableLabel{cnt} = 'on stim adaptive'; 
idxUse = year(pkgRC.timerange(:,1)) == 2020 & ...
        month(pkgRC.timerange(:,1)) == 6 & ... 
        day(pkgRC.timerange(:,1)) == 10 & ... 
        strcmp(pkgRC.side,'L');
pkgData{cnt} = pkgRC(idxUse,:);
cnt = cnt + 1; 
%% apply new labels to PKG data using sliding 7 epoch rule 
% use 3/5 rule to look at 10 minute epochs
for t = 1:length(pkgData)
    pkgTable = pkgData{t};
    for i = 1:size(pkgTable,1)
        if i <5 || i > (size(pkgTable,1)-3)
            pkgTable.new_state{i} = 'state unknown';
        else
            bkvals = pkgTable.BK(i-3:i+3);
            dkvals = pkgTable.DK(i-3:i+3);
            tremrr = pkgTable.Tremor(i-3:i+3);
            statesAssigned = 0;
            bkCutOff = -26;
            if sum(bkvals < bkCutOff & bkvals > -80) >=4 % off
                pkgTable.new_state{i} = 'off';
                statesAssigned = statesAssigned + 1;
            end
            %     Good control (i.e. neither OFF nor ON) is when BKS <26 AND DKS<7
            if sum(bkvals > bkCutOff & dkvals < 7) >=4 % on
                pkgTable.new_state{i} = 'on';
                statesAssigned = statesAssigned + 1;
            end
            %     on with dyskinesia
            if sum(bkvals >= bkCutOff & dkvals >= 7) >=4 % on with dk
                pkgTable.new_state{i} = 'on w dk';
                statesAssigned = statesAssigned + 1;
            end
            %   sleep
            if  sum(bkvals <= -80) >=4 % off
                pkgTable.new_state{i} = 'sleep';
                statesAssigned = statesAssigned + 1;
            end
             %   tremor 
            if  sum(tremrr) >=4 % tremor
                pkgTable.new_state{i} = 'tremor';
                statesAssigned = statesAssigned + 1;
            end
            if statesAssigned  == 0
                pkgTable.new_state{i} = 'state unknown';
            end
            if statesAssigned  > 1
                pkgTable.new_state{i} = 'state rule conflict';
            end
            if max(diff(pkgTable.Date_Time(i-3:i+3))) > minutes(3)
                pkgTable.new_state{i} = 'state unknown';
            end
        end
    end
    pkgData{t} = pkgTable;
end
%% plot cateogries 
clc;
close all;
y = [];
c = [];
for t = 3:4;
    Conditions = categorical(pkgData{t}.new_state,...
        unique(pkgData{1}.new_state));
    fprintf('%s\n',tableLabel{t});
    % new way 
    Cnew  = mergecats(Conditions,{'state unknown','state rule conflict','sleep','tremor'},'sleep-plus');
    summary(Cnew);
    c = countcats(Cnew);
    cats = categories(Cnew);
    y (t,:) = c./sum(c);

    % old way 
%     Conditions = removecats(removecats(Conditions,'state unknown'));
%     Conditions = removecats(removecats(Conditions,'state rule conflict'));
%     Conditions = removecats(removecats(Conditions,'sleep'));
%     Conditions = removecats(removecats(Conditions,'tremor'));
%     idxremove  = isundefined(Conditions);
%     Conditions = Conditions(~idxremove);
%     summary(Conditions);
%     c = countcats(Conditions);
%     cats = categories(Conditions); 
%     y (t,:) = c./sum(c); 
    % 
end
% 

resultdirsave = '/Users/roee/Box/rcs paper paper on first five bilateral implants/revision for nature biotechnology/figures/Fig7.1_new_adaptive';
fnsmv = fullfile(resultdirsave,'process_pkg_data_RCS02_open_loop_vs_closed_loop.mat'); 
filepath = pwd; 
functionname = 'RCS02_open_loop_vs_closed_loop.m';
save(fnsmv,'pkgData','tableLabel','functionname','filepath');


hfig = figure;
hfig.Color = 'w'; 
hbar = bar(y,'stacked');
legend(cats);
hsb = gca;
hsb.XTickLabel = tableLabel;
hsb.XTickLabelRotation = 45; 

%xx 
hsb.XTick = 3:4;
hsb.XTickLabel = tableLabel(3:4);
hsb.XLim = [2 5];
%xx x
ylabel('% time in category'); 
set(gca,'FontSize',16);



% Display one bar for each row of the matrix. The height of each bar is the sum of the elements in the row.
%%
figure;
y = [0.5 0.5 ; 0.1 0.9];
bar(y,'stacked')
legend('1','2');
%%
%

%% plot boxplot histograms of raw data 
addpath(genpath(fullfile(pwd,'toolboxes','Violinplot-Matlab/')));
hfig = figure; 
hfig.Color = 'w'; 
fieldnamesUse = {'BK','DK'};
filenamesLabels = {'Bradykinesia','Dyskinesia'}; 
for f = 1:length(fieldnamesUse)
    hsb(f) = subplot(1,2,f); 
    for t = 1:length(pkgData) 
        varname = sprintf('var%d',t); 
        if strcmp(fieldnamesUse{f},'BK')
            toPlot.(varname) = abs(pkgData{t}.(fieldnamesUse{f}));
        elseif strcmp(fieldnamesUse{f},'DK')
            toPlot.(varname) = log10(abs(pkgData{t}.(fieldnamesUse{f})));
        end
    end
    cla(hsb(f));
    hViolin = violinplot(toPlot);
    title(filenamesLabels{f});
    xlims = hsb(f).XLim;
    if strcmp(fieldnamesUse{f},'BK')
        hp(1) = plot(xlims,[26 26],'LineWidth',2,'Color','r','LineStyle','-.');
        hp(2) = plot(xlims,[80 80],'LineWidth',2,'Color','k','LineStyle','-.');
        ylim([0 200])
        ylabel('Bradykinesia (higher score = more BK)');
    end
    if strcmp(fieldnamesUse{f},'DK')
        hp(1) = plot(xlims,[log10(7) log10(7)],'LineWidth',2,'Color','r','LineStyle','-.');
        hp(2) = plot(xlims,[log10(16) log10(16)],'LineWidth',2,'Color','k','LineStyle','-.');
        ylabel('Dyskinesia (higher score = more DK)');
    end
    hsb(f).XTickLabel = tableLabel;
    hsb(f).XTickLabelRotation = 45;
    hsf(f).FontSize = 16;
    set(gca,'FontSize',16);
end
%%























% open loop 
timeOL = datetime('2020-05-26','Format','uuuu-MM-dd','TimeZone',pkgRC.timerange.TimeZone); 
idxOL = year(pkgRC.timerange(:,1)) == 2020 & ...
        month(pkgRC.timerange(:,1)) == 5 & ... 
        day(pkgRC.timerange(:,1)) == 26;
rcsOL = pkgRC(idxOL,:);

idxOL = year(pkgRC.timerange(:,1)) == 2020 & ...
        month(pkgRC.timerange(:,1)) == 5 & ... 
        day(pkgRC.timerange(:,1)) == 26;
    
% closed loop 
idxCL = year(pkgRC.timerange(:,1)) == 2019 & ...
        month(pkgRC.timerange(:,1)) == 4 & ... 
        day(pkgRC.timerange(:,1)) == 24;
rcsCL = pkgRC(idxCL,:);

% before implant 
idxBI = year(pkgRC.timerange(:,1)) == 2020 & ...
        month(pkgRC.timerange(:,1)) == 6 & ... 
        day(pkgRC.timerange(:,1)) == 10;
rcsBI = pkgRC(idxCL,:);











%% plot histogram - all data 

clear hp
hfig = figure('Color','w','Visible','on');
cnt = 1;

% get rid of NaN data (it's empty on startup
rcsOL = rcsOL(~isnan(rcsOL.BK),:);
rcsCL = rcsCL(~isnan(rcsCL.BK),:);

% get rid of off wrist data
rcsOL = rcsOL(~rcsOL.Off_Wrist,:);
rcsCL = rcsCL(~isnan(rcsCL.BK),:);


% brakdykinesia
hbrdy =  subplot(1,3,cnt); cnt = cnt + 1;
axis(hbrdy);
hold(hbrdy,'on');
hs1 = histogram(rcsOL.BK,'Normalization','probability',...
    'BinWidth',10);
hs2 = histogram(rcsCL.BK,'Normalization','probability',...
    'BinWidth',10);
legend([hs1 hs2],{'open loop', 'closed loop'});
ylims = get(gca,'YLim');
hp(1) = plot([-26 -26],ylims,'LineWidth',2,'Color','r','LineStyle','-.');
hp(2) = plot([-80 -80],ylims,'LineWidth',2,'Color','k','LineStyle','-.');
legend(hp,{'> BK = off','> BK = sleep'});
legend([hs1 hs2],{'open loop', 'closed loop'});
title('BK')
set(gca,'FontSize',16);
clear hp

% tremor
htrem =  subplot(1,3,cnt); cnt = cnt + 1;
axis(htrem);
hold(htrem,'on');
hs1 = histogram(rcsOL.Tremor_Score(rcsOL.Tremor_Score~=0),'Normalization','probability',...
    'BinWidth',5);
hs2 = histogram(rcsCL.Tremor_Score(rcsOL.Tremor_Score~=0),'Normalization','probability',...
    'BinWidth',5);

title('tremor')
set(gca,'FontSize',16);


% dyskinesia
hdysk =  subplot(1,3,cnt); cnt = cnt + 1;
axis(hdysk);
hold(hdysk,'on');
hs1 = histogram(log10(rcsOL.DK),'Normalization','probability',...
    'BinWidth',0.3);
hs2 = histogram(log10(rcsCL.DK),'Normalization','probability',...
    'BinWidth',0.3);


ylims = get(gca,'YLim');
hp(1) = plot([log10(7) log10(7)],ylims,'LineWidth',2,'Color',[0 0.8 0],'LineStyle','-.');
hp(2) = plot([log10(16) log10(16)],ylims,'LineWidth',2,'Color',[0 0.8 0],'LineStyle','-.');

legend(hp(1),{'< DK = dyskinetic'});
title('DK')
set(gca,'FontSize',16);
clear hp

%% loop at violin plots 


clear hp
hfig = figure('Color','w','Visible','on');
cnt = 1;

% get rid of NaN data (it's empty on startup
rcsOL = rcsOL(~isnan(rcsOL.BK),:);
rcsCL = rcsCL(~isnan(rcsCL.BK),:);

% get rid of off wrist data
rcsOL = rcsOL(~rcsOL.Off_Wrist,:);
rcsCL = rcsCL(~isnan(rcsCL.BK),:);


% brakdykinesia
hbrdy =  subplot(1,3,cnt); cnt = cnt + 1;
axis(hbrdy);
hold(hbrdy,'on');


toplot = {};
idxNotAsleepOL = ~(rcsOL.BK <= -80); 
idxNotAsleepCL = ~(rcsCL.BK <= -80); 
toplot{1,1} = abs(rcsOL.BK(idxNotAsleepOL));
toplot{1,2} = abs(rcsCL.BK(idxNotAsleepCL));
hviolin  = violin(toplot);
hviolin(1).FaceColor = [0.5 0.5 0.5];
hviolin(1).FaceAlpha = 0.3;

hviolin(2).FaceColor = [0 0.8 0];
hviolin(2).FaceAlpha = 0.3;

hbrdy.XTick = [1 2]; 
hbrdy.XTickLabel = {'open loop','closed loop'}; 
hbrdy.XTickLabelRotation = 45; 

% legend([hviolin(1) hviolin(2)],{'open loop', 'closed loop'});
title('Bradykinesia')
set(gca,'FontSize',16);
clear hp






% tremor
htrem =  subplot(1,3,cnt); cnt = cnt + 1;
axis(htrem);
hold(htrem,'on');

toplot = {};
toplot{1,1} = rcsOL.Tremor_Score( (rcsOL.Tremor_Score ~= 0) & idxNotAsleepOL);
toplot{1,2} = rcsCL.Tremor_Score((rcsCL.Tremor_Score ~= 0) & idxNotAsleepCL);
hviolin  = violin(toplot);
hviolin(1).FaceColor = [0.5 0.5 0.5];
hviolin(1).FaceAlpha = 0.3;

hviolin(2).FaceColor = [0 0.8 0];
hviolin(2).FaceAlpha = 0.3;

htrem.XTick = [1 2]; 
htrem.XTickLabel = {'open loop','closed loop'}; 
htrem.XTickLabelRotation = 45; 

% legend([hviolin(1) hviolin(2)],{'open loop', 'closed loop'});
title('Tremor')
set(gca,'FontSize',16);
clear hp




% dyskinesia
hdysk =  subplot(1,3,cnt); cnt = cnt + 1;
axis(hdysk);
hold(hdysk,'on');

toplot = {};

DKOL = log10(rcsOL.DK);
DKOL = DKOL(idxNotAsleepOL); 

DKCL = log10(rcsCL.DK);
DKCL = DKCL(idxNotAsleepCL); 

toplot{1,1} = DKOL; 
toplot{1,2} = DKCL;
hviolin  = violin(toplot);
hviolin(1).FaceColor = [0.5 0.5 0.5];
hviolin(1).FaceAlpha = 0.3;

hviolin(2).FaceColor = [0 0.8 0];
hviolin(2).FaceAlpha = 0.3;

hdysk.XTick = [1 2]; 
hdysk.XTickLabel = {'open loop','closed loop'}; 
hdysk.XTickLabelRotation = 45; 

% legend([hviolin(1) hviolin(2)],{'open loop', 'closed loop'});
title('Dyskinesia')
set(gca,'FontSize',16);
clear hp

%% sort data a bit more - plot only not asleep data between 
%% 9am - 10pm. 
%% also create 10 minute scores 
close all;
hfig = figure; 
hfig.Color = 'w'; 
hsb(1) = subplot(2,2,1); 

% OL
states = rcsOL.states;
Conditions = categorical(states(3:end),...
    unique(states(3:end)));
h = histogram(Conditions,'Normalization','probability');
h.DisplayOrder = 'descend';
ylabel('% time / condition');
titleUse{1,1} = 'RCS02 open loop';
titleUse{1,2} = 'all states - 10 minute estimate';
title(titleUse)
hsb(1).XTickLabelRotation = 45; 
set(gca,'FontSize',16);


% CL
hsb(2) = subplot(2,2,2); 
states = rcsCL.states;
Conditions = categorical(states(3:end),...
    unique(states(3:end)));
h = histogram(Conditions,'Normalization','probability');
h.DisplayOrder = 'descend';
ylabel('% time / condition');
titleUse{1,1} = 'RCS02 adaptive DBS';
titleUse{1,2} = 'all states - 10 minute estimate';
title(titleUse)
hsb(2).XTickLabelRotation = 45; 
set(gca,'FontSize',16);

% plot comparison 
statesCompare = {'off','on','tremor','dyskinesia severe','sleep'};
statesRaw{1} = rcsOL.states; 
statesRaw{2} = rcsCL.states; 

for a = 1:2 % open loop (1) vs closed loop (2) 
    for s = 1:length(statesCompare)
        idxcond = cellfun(@(x) strcmp(x,statesCompare{s}),statesRaw{a});
        percentStates(s,a) = sum(idxcond)/length(statesRaw{a});
    end
end



hsb(3) = subplot(2,2,3); 
bar(percentStates);
legend({'open loop','closed loop'});
hsb(3).XTickLabel = statesCompare; 
hsb(3).XTickLabelRotation = 45; 
titleUse{1,1} = 'RCS02 adaptive DBS';
titleUse{1,2} = '% reduction in some PKG state estimates (10 min)';
title(titleUse)

set(gca,'FontSize',16);

% more direct comparison between key metrics 
hsb(4) = subplot(2,2,4); 
percOver = percentStates(:,2)./percentStates(:,1);
bar(percOver);
Y = percOver';
text(1:length(Y),Y,num2str(Y'),'vert','bottom','horiz','center'); 
box off

titleUse{1,1} = 'RCS02 adaptive DBS';
titleUse{1,2} = 'closed loop / open loop %';
hsb(4).XTickLabel = statesCompare; 
hsb(4).XTickLabelRotation = 45; 
title(titleUse)
set(gca,'FontSize',16);









