function set_env
%% figure out user directory

if ispc
    userDir = winqueryreg('HKEY_CURRENT_USER',...
        ['Software\Microsoft\Windows\CurrentVersion\' ...
        'Explorer\Shell Folders'],'Personal');
    userDir = fileparts(userDir);
else
    userDir = char(java.lang.System.getProperty('user.home'));
end

%% template for setting environment
if isempty(which('dnc_set_env'))
    % Defaults
    project = 'null_analysis';
    setenv('D_USER', userDir);
    setenv('D_PROJECT', fullfile('D:', project));
    setenv('D_DATA_IN', fullfile(getenv('D_PROJECT'), 'data_physio'));
    setenv('D_DATA_OUT', fullfile(getenv('D_PROJECT'), 'proc_data'));
    setenv('STR_DATA_TYPE', 'neural data');
    setenv('STR_DATA_FOLDER', 'Combined');
    setenv('D_PROC_DATA', fullfile(getenv('D_PROJECT'), 'proc_pass'));
    setenv('D_FIGURE', fullfile(getenv('D_PROJECT'), 'figures'));
    setenv('D_ANALYZE_RCS', '');
    setenv('D_SPM', '');
    setenv('D_LEADDBS', '');
else
    dnc_set_env(userDir);
end

%% recursive add commands

% add_this_path_recursive(getenv('D_TDT_SDK'));
% add_this_path_recursive(fullfile(getenv('D_GIT'),'auxf'));

%% Insert into dnc_set_env.m as needed!
% slce = slCharacterEncREPORTSoding;
% slce_target = 'UTF-8';
% if not(strcmpi(slce, slce_target))
%     slCharacterEncoding(slce_target);
% end
end

%% separate function def at the end

function add_this_path_recursive(p)
current_path_contents = path;
current_path_contents = string(strsplit(current_path_contents, ';'));
if not(any(current_path_contents==p))
    addpath(genpath(p));
end
end

%% template for dnc_set_env
% function dnc_set_env(userDir)
% if exist(userDir, 'dir')==7
%     % Defaults
%     project = 'motor_mi_decode';
%     setenv('D_USER', userDir);
%     setenv('D_PROJECT', fullfile(getenv('D_USER'), 'local', 'data', 'gangulylab'));
%     setenv('D_DATA', fullfile(getenv('D_PROJECT'), 'raw_data'));
%     setenv('D_PROC_DATA', fullfile(getenv('D_PROJECT'), 'proc_data'));
%     setenv('D_FIGURE', fullfile(getenv('D_PROJECT'), 'figures'));
% end
% end