%% environment setup

clearvars -except nothing;
clc;
close all;

% add relative path from current repo
addpath(fullfile('..'))
addpath(fullfile('..', 'preproc'))
addpath(fullfile('..', 'utils'))
addpath(fullfile('..', 'plotting'))

% set envrionment
set_env;
addpath(fullfile(getenv('D_ANALYZE_RCS'), 'code'));
addpath(fullfile(getenv('D_SPM')));
addpath(fullfile(getenv('D_LEADDBS')));


%% now start leadDBS

% start lead DBS
lead

t1= 1;

