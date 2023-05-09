function dnc_set_env(userDir, varargin)
%% Input parsing
% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'bool_aDBS', true, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
bool_aDBS = p.Results.bool_aDBS;                                            % boolean flag for loading accel data also

%% setting the envrionment

if exist(userDir, 'dir')==7
    %% common parameters
    % direct path setting
    setenv('D_USER', userDir);
    setenv('D_PROJECT', fullfile(getenv('D_USER'), 'local', 'data', 'starrlab'));
    setenv('D_PROC_DATA', fullfile(getenv('D_PROJECT'), 'proc_data'));
    setenv('D_DATA_OUT', fullfile(getenv('D_PROJECT'), 'proc_data'));
    setenv('D_FIGURE', fullfile(getenv('D_PROJECT'), 'figures'));

    % external package path setting
    setenv('D_ANALYZE_RCS', fullfile(getenv('D_USER'), 'local', ...
        'gitprojects', 'Analysis-rcs-data'));

    % folder naming setting                              
    setenv('STR_DATA_TYPE', 'neural data');                                 % second organization level: neural data vs PKG vs AW
    setenv('STR_DATA_FOLDER', 'Combined');                                  % third organization level: aDBS vs SCBS

    %% Defaults for aDBS based analysis
    if bool_aDBS
        % input data path
        setenv('D_DATA_IN', fullfile(getenv('D_PROJECT'), ...               % first organization level: aDBS folder as on box
            'Structured_aDBS_pipeline', 'Data')); 
        
        % name of aDBS specific sub-folders
        setenv('STR_STEP3', 'Step3_in_clinic_neural_recordings');
        setenv('STR_STEP4', 'Step4_at_home_neural_recordings');
        setenv('STR_STEP5', 'Step5_supervised_aDBS');
        setenv('STR_STEP6', 'Step6_at_home aDBS_short');
        setenv('STR_STEP7', 'Step7_at_home aDBS_long');
        
    %% Defaults for PAC based analysis
    else
        % input data path
        setenv('D_DATA_IN', fullfile(getenv('D_PROJECT'), ...               % first organization level: raw data folder for pre-adaptive data
            'raw_data'));
    end

end
end