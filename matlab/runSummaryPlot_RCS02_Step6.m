%% Envrionment setup

clearvars -except nothing;
clc;
close all;

set_env;
addpath(fullfile(getenv('D_ANALYZE_RCS'), 'code'));
addpath(genpath(fullfile('..', 'auxf', 'external')));
addpath(genpath(fullfile('..', 'auxf', 'internal/')));

addpath(fullfile('preproc'))
addpath(fullfile('utils'))
addpath(fullfile('plotting'))


%% obtain all data of interest


% vec_pfe_summary_output = glob(fullfile('/home/jyao/local/data/starrlab/proc_data/RCS02/Structured_aDBS_pipeline/Step6_at_home aDBS_short/summaryStats_round12_R/',...
%     'summary*'));

vec_pfe_summary_output = glob(fullfile('/home/jyao/local/data/starrlab/proc_data/RCS02/Structured_aDBS_pipeline/Step6_at_home aDBS_short/summaryStats_round3_R/',...
    'summary*'));


for i = 1:numel(vec_pfe_summary_output)
    t1 = load(vec_pfe_summary_output{i});
    vecSummaryOutput{i} = t1.summaryOutput; 
end

% xlimLowDate = datetime(2022, 9 , 13);
% xlimHighDate = datetime(2022, 9 , 29);

xlimLowDate = datetime(2022, 10 , 23);
xlimHighDate = datetime(2022, 11 , 02);

% vecDateTime = [datetime(2022, 9 , 14); ...
%     datetime(2022, 9 , 16);
%     datetime(2022, 9 , 18);
%     datetime(2022, 9 , 20);
%     datetime(2022, 9 , 22);
%     datetime(2022, 9 , 25);
%     datetime(2022, 9 , 27);
%     datetime(2022, 9 , 28)];

vecDateTime = [datetime(2022, 10 , 24); ...
    datetime(2022, 10 , 26);
    datetime(2022, 10 , 28);
    datetime(2022, 10 , 30);
    datetime(2022, 11 , 01)];



xTicks = vecDateTime;
% vecStr_xTicks = {'\color{blue} 2022/09/14 Beta';
%     '\color{black} 2022/09/16 cDBS';
%     '\color{red} 2022/09/18 Combo';
%     '\color{orange} 2022/09/20 Gamma';
%     '\color{red} 2022/09/22 Combo';
%     '\color{blue} 2022/09/25 Beta';
%     '\color{black} 2022/09/27 cDBS';
%     '\color{orange} 2022/09/28 Gamma'};

vecStr_xTicks = {'\color{orange} 2022/10/24 Gamma';
    '\color{black} 2022/10/26 cDBS';
    '\color{blue} 2022/10/28 Beta';
    '\color{red} 2022/10/30 Combo';
    '\color{orange} 2022/11/01 Gamma'};

% current related variables
vecAvgCurrentIn = [];
vecStdCurrentIn = [];

vecPerMD_ON = [];
vecPerMD_OFF = [];

vecPerMDBrady = [];
vecPerMDDysk = [];

vecPerBradyTbsome = [];
vecPerDyskTbsome = [];

vecTotalDyskTime = [];

vecPerAWDysk = [];
vecAvgAWDysk = [];

vecAvgMDBrady = [];
vecStdMDBrady = [];
vecAvgMDDysk = [];
vecStdMDDysk = [];

vecAvgPKGBrady = [];
vecStdPKGBrady = [];
vecAvgPKGDysk = [];
vecStdPKGDysk = [];

% On and OFF specific wearables
vecOnAWDyskFull = [];
vecOffAWDyskFull = [];

vecOnPKGDyskFull = [];
vecOffPKGDyskFull = [];

vecOnPKGBradyFull = [];
vecOffPKGBradyFull = [];

for i = 1:numel(vecSummaryOutput)
    vecAvgCurrentIn = [vecAvgCurrentIn; vecSummaryOutput{i}.avgCurrentIn];
    vecStdCurrentIn = [vecStdCurrentIn; vecSummaryOutput{i}.stdCurrentIn];

    vecPerMD_ON = [vecPerMD_ON; vecSummaryOutput{i}.perMD_ON];
    vecPerMD_OFF = [vecPerMD_OFF; vecSummaryOutput{i}.perMD_OFF];

    vecPerMDBrady = [vecPerMDBrady; vecSummaryOutput{i}.perMDBrady];
    vecPerMDDysk = [vecPerMDDysk; vecSummaryOutput{i}.perMDDysk];

    vecPerBradyTbsome = [vecPerBradyTbsome; vecSummaryOutput{i}.perMDBradyTbsome];
    vecPerDyskTbsome = [vecPerDyskTbsome; vecSummaryOutput{i}.perMDDyskTbsome];
    
    if ~isempty(vecSummaryOutput{i}.allMDBrady)
        vecAvgMDBrady = [vecAvgMDBrady; mean(vecSummaryOutput{i}.allMDBrady)];
        vecStdMDBrady = [vecStdMDBrady; std(vecSummaryOutput{i}.allMDBrady)];
    else
        vecAvgMDBrady = [vecAvgMDBrady; 0];
        vecStdMDBrady = [vecStdMDBrady; 0];
    end
        
    if ~isempty(vecSummaryOutput{i}.allMDDysk)
        vecAvgMDDysk = [vecAvgMDDysk; mean(vecSummaryOutput{i}.allMDDysk)];
        vecStdMDDysk = [vecStdMDDysk; std(vecSummaryOutput{i}.allMDDysk)];
    else
        vecAvgMDDysk = [vecAvgMDDysk; 0];
        vecStdMDDysk = [vecStdMDDysk; 0];
    end

    vecTotalDyskTime = [vecTotalDyskTime; vecSummaryOutput{i}.totalDyskTime];
    
    
    % optionally append AW data if exists
    if any(strcmp('perAWDysk', fieldnames(vecSummaryOutput{i})))
        vecPerAWDysk = [vecPerAWDysk; vecSummaryOutput{i}.perAWDysk];
        vecAvgAWDysk = [vecAvgAWDysk; vecSummaryOutput{i}.avgAWDysk];
        vecOnAWDyskFull = [vecOnAWDyskFull; vecSummaryOutput{i}.onAWDysk];
        vecOffAWDyskFull = [vecOffAWDyskFull; vecSummaryOutput{i}.offAWDysk];
    else
        vecPerAWDysk = [vecPerAWDysk; NaN];
        vecAvgAWDysk = [vecAvgAWDysk; NaN];
    end

    % optionally append PKG data if exists
    if any(strcmp('allPKGBrady', fieldnames(vecSummaryOutput{i})))
        currBrady = vecSummaryOutput{i}.allPKGBrady;
        currDysk = vecSummaryOutput{i}.allPKGDysk;

        vecAvgPKGBrady = [vecAvgPKGBrady; ...
            mean(currBrady(~isinf(currBrady)))];
        vecStdPKGBrady = [vecStdPKGBrady; ...
            std(currBrady(~isinf(currBrady)))];
        vecAvgPKGDysk = [vecAvgPKGDysk; ...
            mean(currDysk(~isinf(currDysk)))];
        vecStdPKGDysk = [vecStdPKGDysk; ...
            std(currDysk(~isinf(currDysk)))];

        vecOnPKGDyskFull = [vecOnPKGDyskFull; vecSummaryOutput{i}.onPKGDysk];
        vecOffPKGDyskFull = [vecOffPKGDyskFull; vecSummaryOutput{i}.offPKGDysk];

        vecOnPKGBradyFull = [vecOnPKGBradyFull; vecSummaryOutput{i}.onPKGBrady];
        vecOffPKGBradyFull = [vecOffPKGBradyFull; vecSummaryOutput{i}.offPKGBrady];
    else
        vecAvgPKGBrady = [vecAvgPKGBrady; NaN];
        vecStdPKGBrady = [vecStdPKGBrady; NaN];
        vecAvgPKGDysk = [vecAvgPKGDysk; NaN];
        vecStdPKGDysk = [vecStdPKGBrady; NaN];
    end
end

% %%
% 
% % plot a boxplot with all the wearable data
% 
% gAWDyskOn = repmat({'AWDysk On'},size(vecOnAWDyskFull, 1), 1);
% gAWDyskOff = repmat({'AWDysk Off'},size(vecOffAWDyskFull, 1), 1);
% 
% gPKGDyskOn = repmat({'PKGDysk On'},size(vecOnPKGDyskFull, 1), 1);
% gPKGDyskOff = repmat({'PKGDysk Off'},size(vecOffPKGDyskFull, 1), 1);
% 
% gPKGBradyOn = repmat({'PKGBrady On'},size(vecOnPKGDyskFull, 1), 1);
% gPKGBradyOff = repmat({'PKGBrady Off'},size(vecOffPKGDyskFull, 1), 1);
% 
% g1 = [gAWDyskOn; gAWDyskOff];
% g2 = [gPKGDyskOn; gPKGDyskOff];
% g3 = [gPKGBradyOn; gPKGBradyOff];
% 
% b_AWDysk = [vecOnAWDyskFull; vecOffAWDyskFull];
% b_PKGDysk = [vecOnPKGDyskFull; vecOffPKGDyskFull];
% b_PKGBrady = [vecOnPKGBradyFull; vecOffPKGBradyFull];
% 
% boxplot(b_AWDysk, g1)
% ylim([-0.1, 1.1]);
% title("AW Dysk scores")
% 
% boxplot(b_PKGDysk, g2)
% ylim([-1, 30]);
% title("PKG Dysk scores")
% 
% 
% 
% boxplot(b_PKGBrady, g3)
% % ylim([-5, 30]);
% 
% title("PKG Brady scores")


%% plot changes over time and with stim

figure();
ax1 = subplot(611); hold on;
plot(vecDateTime, vecPerMD_ON, 'Marker', 's', ...
    'DisplayName', 'MD: Per ON')
plot(vecDateTime, vecPerMD_OFF, 'Marker', 's', ...
    'DisplayName', 'MD: Per OFF')
ylim([-0.2, 1.2]); ylabel('Percentage')
title('Percentage of ON and OFF periods in MD')
legend('boxoff');
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

ax2 = subplot(612); hold on;
plot(vecDateTime, vecPerMDDysk, 'Marker', 's', ...
    'DisplayName', 'MD: Per Dysk')
plot(vecDateTime, vecPerMDBrady, 'Marker', 's', ...
    'DisplayName', 'MD: Per Brady')
ylim([-0.2, 1.2]); ylabel('Percentage')
title('Percentage of Dysk and Brady periods in MD')
legend('boxoff');
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

ax3 = subplot(613); hold on;
plot(vecDateTime, vecPerDyskTbsome, 'Marker', 's', ...
    'DisplayName', 'MD: Per Tbsome Dysk')
plot(vecDateTime, vecPerBradyTbsome, 'Marker', 's', ...
    'DisplayName', 'MD: Per Tbsome Brady')
ylim([-0.2, 1.2]); ylabel('Percentage')
title('Percentage of Troublesome Dysk and Brady periods in MD')
legend('boxoff');
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

ax4 = subplot(614); hold on;
errorbar(vecDateTime, vecAvgMDDysk, vecStdMDDysk, 'Marker', 's', ...
    'DisplayName', 'MD: Avg Dysk Intensity')
errorbar(vecDateTime, vecAvgMDBrady, vecStdMDBrady, 'Marker', 's', ...
    'DisplayName', 'MD: Avg Brady Intensity')
ylim([-0.2, 3.2]); ylabel('Avg Intensity (a.u.)')
title('Average Intensity of Dysk and Brady')
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

ax5 = subplot(615); hold on;
plot(vecDateTime, vecTotalDyskTime, 'Marker', 's')
title('Duration of self reported Dysk episodes');
ylabel('Duration (min)')
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

ax6 = subplot(616); hold on;
errorbar(vecDateTime, vecAvgCurrentIn, vecStdCurrentIn, 'Marker', 's')
ylim([2.8, 3.8]); ylabel('Current Injected');
title('Average and Standard Deviation of Current Injected')
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

linkaxes([ax1, ax2, ax3, ax4, ax5, ax6], 'x');
xlim([xlimLowDate, xlimHighDate]);

%%

figure();
ax1 = subplot(511); hold on;
plot(vecDateTime, vecPerAWDysk, 'Marker', 's')
title('Percentage of AW Dysk Periods')
ylabel('Percentage');
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

ax2 = subplot(512); hold on;
plot(vecDateTime, vecAvgAWDysk, 'Marker', 's')
title('Average AW Dysk Probability')
ylabel('AW Dysk Probability');
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

ax3 = subplot(513); hold on;
errorbar(vecDateTime, vecAvgPKGDysk, vecStdPKGDysk, 'Marker', 's')
title('Average PKG Dysk Score')
ylabel('PKG Dysk Score'); ylim([-10, 20])
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

ax4 = subplot(514); hold on;
errorbar(vecDateTime, vecAvgPKGBrady, vecStdPKGBrady, 'Marker', 's')
title('Average PKG Brady Score')
ylabel('PKG Brady Score'); ylim([0, 80])
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

ax5 = subplot(515); hold on;
errorbar(vecDateTime, vecAvgCurrentIn, vecStdCurrentIn, 'Marker', 's')
ylim([2.8, 3.8]); ylabel('Current Injected');
title('Average and Standard Deviation of Current Injected')
xticks(vecDateTime);
xticklabels(vecStr_xTicks);

linkaxes([ax1, ax2, ax3, ax4, ax5], 'x');
xlim([xlimLowDate, xlimHighDate]);

%% compare PKG score distribution

PKG_Dysk_cDBS = [vecSummaryOutput{2}.allPKGDysk; ...
    vecSummaryOutput{7}.allPKGDysk];
PKG_Dysk_Beta = [vecSummaryOutput{1}.allPKGDysk; ...
    vecSummaryOutput{6}.allPKGDysk];
PKG_Dysk_Gamma = [vecSummaryOutput{4}.allPKGDysk; ...
    vecSummaryOutput{8}.allPKGDysk];
PKG_Dysk_Combo = [vecSummaryOutput{4}.allPKGDysk; ...
    vecSummaryOutput{8}.allPKGDysk];

PKG_Brady_cDBS = [vecSummaryOutput{2}.allPKGBrady; ...
    vecSummaryOutput{7}.allPKGBrady];
PKG_Brady_Beta = [vecSummaryOutput{1}.allPKGBrady; ...
    vecSummaryOutput{6}.allPKGBrady];
PKG_Brady_Gamma = [vecSummaryOutput{4}.allPKGBrady; ...
    vecSummaryOutput{8}.allPKGBrady];
PKG_Brady_Combo = [vecSummaryOutput{4}.allPKGBrady; ...
    vecSummaryOutput{8}.allPKGBrady];


g1 = repmat({'cDBS'},size(PKG_Dysk_cDBS, 1), 1);
g2 = repmat({'Beta'},size(PKG_Dysk_Beta, 1),1);
g3 = repmat({'Gamma'},size(PKG_Dysk_Gamma, 1),1);
g4 = repmat({'Combo'},size(PKG_Dysk_Combo, 1),1);
g = [g1; g2; g3; g4];

b_Dysk = [PKG_Dysk_cDBS; PKG_Dysk_Beta; PKG_Dysk_Gamma; PKG_Dysk_Combo];
b_Brady = [PKG_Brady_cDBS; PKG_Brady_Beta; PKG_Brady_Gamma; PKG_Brady_Combo];

boxplot(b_Dysk, g)
ylim([-1, 10]);
title("PKG Dysk scores")

boxplot(b_Brady, g);
title("PKG Brady scores")
