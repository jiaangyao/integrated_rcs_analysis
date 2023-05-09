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

%% setting some important flags

BOOL_STN_COMP = 1;
BOOL_RAW_FILT = 0;
IDX_POW_CH = 1;
IDX_TD_CH = 1;
STR_SIDE = 'R';

%% obtain all data of interest

% obtain the round 3 dates
p_summary_base = '/home/jyao/local/data/starrlab/proc_data/RCS02/Structured_aDBS_pipeline/Step6_at_home aDBS_short/';
vec_pfe_day_round3 = glob(fullfile(p_summary_base, 'third round_orig/', ...
    '2022*'));

% obtain the round 3 data per day
[vec_str_day_round3, vec_pow_data_day_round3, vec_td_data_day_round3] = ...
        load_session(vec_pfe_day_round3, STR_SIDE, BOOL_STN_COMP, ...
        BOOL_RAW_FILT, IDX_POW_CH, IDX_TD_CH);

% obtain the round 4 dates
vec_pfe_day_round4 = glob(fullfile(p_summary_base, 'fourth round_mod/', ...
    '2022*'));
[vec_str_day_round4, vec_pow_data_day_round4, vec_td_data_day_round4] = ...
        load_session(vec_pfe_day_round4, STR_SIDE, BOOL_STN_COMP, ...
        BOOL_RAW_FILT, IDX_POW_CH, IDX_TD_CH);

% vecStr_xTicks = {'\color{orange} 20221024 Beta',...
%     '\color{black} 20221026 cDBS',...
%     '\color{black} 20221028 cDBS',...
%     '\color{black} 20221030 cDBS',...
%     '\color{black} 20221101 cDBS',...
%     '\color{orange} 20221114 Beta',...
%     '\color{orange} 20221116 Beta',...
%     '\color{orange} 20221118 Beta',...
%     '\color{black} 20221120 cDBS',...
%     '\color{orange} 20221121 Beta',...
%     '\color{orange} 20221123 Beta'};

vecStr_xTicks = {'\color{orange} 20221024 Gamma',...
    '\color{black} 20221026 cDBS',...
    '\color{black} 20221028 Beta',...
    '\color{black} 20221030 Combo',...
    '\color{black} 20221101 Gamma',...
    '\color{orange} 20221114 Combo',...
    '\color{orange} 20221116 Combo',...
    '\color{orange} 20221118 Combo',...
    '\color{black} 20221120 cDBS',...
    '\color{orange} 20221121 Combo',...
    '\color{orange} 20221123 Combo/Gamma'};

%% plot the PSDs

fig = figure; 
for i = 1:numel(vec_str_day_round4)
    td_data_day_curr = vec_td_data_day_round4{i};
    str_day_curr = vec_str_day_round4{i};

    % now calculate pwelch
    [pxxCurr, fCurr] = pwelch(td_data_day_curr, 2^nextpow2(250), [], [], 250);

    % plot the PSD
    semilogy(fCurr, pxxCurr, 'LineWidth', 2, 'DisplayName', str_day_curr);
    hold on;
end

legend('boxoff')
xlim([0, 100])

%% now plot the violin plot

xlabel = [vec_str_day_round3, vec_str_day_round4];
vec_pow_data = [vec_pow_data_day_round3, vec_pow_data_day_round4];

figure;
boxplotGroup(vec_pow_data, 'primarylabels', xlabel);

set(gca, 'XTickLabel', vecStr_xTicks)
% violin(vec_pow_data, 'xlabel', xlabel);
ylim([-100, 5000]);


ylabel('Power (A.U.)');
title('Power Distribution Across Days')
xticklabels(vecStr_xTicks);

%% now plot the time domain stuff

%
vec_td_data = [vec_td_data_day_round3, vec_td_data_day_round4];
boxplotGroup(vec_td_data, 'primarylabels', xlabel);

set(gca, 'XTickLabel', vecStr_xTicks)
% violin(vec_pow_data, 'xlabel', xlabel);
ylim([-500, 500]);


ylabel('Amplitude (A.U.)');
title('Time Domain Distribution Across Days')
xticklabels(vecStr_xTicks);

%%

function [vec_str_day, vec_pow_data_day, vec_td_data_day] = ...
    load_session(vec_pfe_day, STR_SIDE, BOOL_STN_COMP, BOOL_RAW_FILT, ...
    IDX_POW_CH, IDX_TD_CH)

vec_str_day = {};
vec_pow_data_day = {};
vec_td_data_day = {};

for i = 1:numel(vec_pfe_day)
    % obtain current date
    pfe_day_curr = vec_pfe_day{i};
    str_day_curr = split(pfe_day_curr, '/');
    str_day_curr = str_day_curr{numel(str_day_curr) - 1};
    vec_str_day{i} = str_day_curr;

    % now glob processed sessions
    vec_pfe_proc_data_curr = glob(fullfile(pfe_day_curr, ['RCS02', STR_SIDE ], ...
        'raw_struct/', 'raw_struct*'));
    pow_data_day = [];
    td_data_day = [];
    for j = 1:numel(vec_pfe_proc_data_curr)
        % loading processed session
        pfe_proc_data_curr = vec_pfe_proc_data_curr{j};
        fprintf('Loading %s...\n', pfe_proc_data_curr)
        load(pfe_proc_data_curr);

        if BOOL_STN_COMP
            pow_ch_curr = raw_struct_curr.pow_data_stn(:, IDX_POW_CH);
        else
            pow_ch_curr = raw_struct_curr.pow_data_motor(:, IDX_POW_CH);
        end

        % also get a sense of the range of time domain
        if BOOL_RAW_FILT
            raw_ch_curr = raw_struct_curr.raw_data_filt(:, IDX_TD_CH);
        else
            raw_ch_curr = raw_struct_curr.raw_data(:, IDX_TD_CH);
        end
        
        % append to outer loop
        pow_data_day = [pow_data_day; pow_ch_curr];
        td_data_day = [td_data_day; raw_ch_curr];
    end
    fprintf('\n')

    % now compute statistics
    vec_pow_data_day{i} = pow_data_day;
    vec_td_data_day{i} = td_data_day;

end

end