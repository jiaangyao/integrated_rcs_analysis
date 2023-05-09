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


%% Define the font size

% set various font sizes
titleFontSize = 16;
labelFontSize = 16;
legendFontSize = 13;
tickFontSize = 13;

%% Load the summary structure

vec_pfe_struct_20221114 = glob(fullfile('/home/jyao/local/data/starrlab/proc_data/RCS02/Structured_aDBS_pipeline/Step6_at_home aDBS_short/20221114/',...
    'raw_struct*'));

vec_pfe_struct_20221116 = glob(fullfile('/home/jyao/local/data/starrlab/proc_data/RCS02/Structured_aDBS_pipeline/Step6_at_home aDBS_short/20221116/',...
    'raw_struct*'));

vec_pfe_struct_20221118 = glob(fullfile('/home/jyao/local/data/starrlab/proc_data/RCS02/Structured_aDBS_pipeline/Step6_at_home aDBS_short/20221118/',...
    'raw_struct*'));

vec_pfe_struct_20221120 = glob(fullfile('/home/jyao/local/data/starrlab/proc_data/RCS02/Structured_aDBS_pipeline/Step6_at_home aDBS_short/20221120/',...
    'raw_struct*'));

% now load the 20221114 data
powDataFullDay1 = [];
for i = 1:numel(vec_pfe_struct_20221114)
    pfeStructCurr = vec_pfe_struct_20221114{i};
    structCurr = load(pfeStructCurr).raw_struct_curr;
    powDataFullDay1 = [powDataFullDay1; structCurr.pow_data_stn];
end

% then load the 20221116 data
powDataFullDay2 = [];
for i = 1:numel(vec_pfe_struct_20221116)
    pfeStructCurr = vec_pfe_struct_20221116{i};
    structCurr = load(pfeStructCurr).raw_struct_curr;
    powDataFullDay2 = [powDataFullDay2; structCurr.pow_data_stn];
end

% then load the 20221118 data
powDataFullDay3 = [];
for i = 1:numel(vec_pfe_struct_20221118)
    pfeStructCurr = vec_pfe_struct_20221118{i};
    structCurr = load(pfeStructCurr).raw_struct_curr;
    powDataFullDay3 = [powDataFullDay3; structCurr.pow_data_stn];
end


% then load the 20221120 data
powDataFullDay4 = [];
for i = 1:numel(vec_pfe_struct_20221120)
    pfeStructCurr = vec_pfe_struct_20221120{i};
    structCurr = load(pfeStructCurr).raw_struct_curr;
    powDataFullDay4 = [powDataFullDay4; structCurr.pow_data_stn];
end


%% now try to do the histogram of the distribution of the power

powDataSTNDay1 = powDataFullDay1(:, 3);
powDataSTNDay2 = powDataFullDay2(:, 3);
powDataSTNDay3 = powDataFullDay3(:, 3);
powDataSTNDay4 = powDataFullDay4(:, 3);

% now normalize nbins by bin size
BIN_SIZE = 100;
nbinDay1 = round(max(powDataSTNDay1) / BIN_SIZE);
nbinDay2 = round(max(powDataSTNDay2) / BIN_SIZE);
nbinDay3 = round(max(powDataSTNDay3) / BIN_SIZE);
nbinDay4 = round(max(powDataSTNDay4) / BIN_SIZE);

% now do the histogram
figure;
subplot(411)
h1 = histogram(powDataSTNDay1, nbinDay1, 'DisplayName', '20221114');
ax = gca;
ax.FontSize = tickFontSize;
xlim([-500, 5e3])
% add legend
lgdCurr = legend('boxoff');
lgdCurr.FontSize;
ylabel('Count', 'FontSize', labelFontSize)

subplot(412)
h2 = histogram(powDataSTNDay2, nbinDay2, 'DisplayName', '20221116');
ax = gca;
ax.FontSize = tickFontSize;
xlim([-500, 5e3])
% add legend
lgdCurr = legend('boxoff');
lgdCurr.FontSize;
ylabel('Count', 'FontSize', labelFontSize)

subplot(413)
h3 = histogram(powDataSTNDay3, nbinDay3, 'DisplayName', '20221118');
ax = gca;
ax.FontSize = tickFontSize;
xlim([-500, 5e3])
% add legend
lgdCurr = legend('boxoff');
lgdCurr.FontSize;
ylabel('Count', 'FontSize', labelFontSize)

subplot(414)
h4 = histogram(powDataSTNDay4, nbinDay4, 'DisplayName', '20221120');
ax = gca;
ax.FontSize = tickFontSize;
xlim([-500, 5e3])
% add legend
lgdCurr = legend('boxoff');
lgdCurr.FontSize;
ylabel('Count', 'FontSize', labelFontSize)

% add title
title('STN Beta Power For LD Across the Days', 'FontSize', titleFontSize)

% label and change the limit
xlabel('STN Beta Power (A.U.)', 'FontSize', labelFontSize)



