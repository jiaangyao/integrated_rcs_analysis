function plotLogsOverTime(pFigure, vec_output, pkgWatchTable, ...
    str_on_off_meds, str_side, str_data_day, cfg_in, varargin)
%% Input parsing
% adding custom input flag so can implement new path searching without
% modifying existing scripts - JY 07/18/2022

% Handle the optional inputs
p = inputParser;
p.KeepUnmatched = true;

addParameter(p, 'boolSaveAsFig', false, ...
    @(x) validateattributes(x, {'logical'}, {'nonempty'}));

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
boolSaveAsFig = p.Results.boolSaveAsFig;

% Define all magic numbers
N_SMOOTHING_FILTER_LENGTH_SEC = 5;


%% unpack data

cfg = cfg_in;

% loop through all sessions
vecTime = [];
vecState = [];
vecAmp = [];
vecAliasedState = [];

for idx_sess = 1:numel(vec_output)
    % unpack the variables
    output_curr = vec_output{idx_sess};

    % now pack to outer structure
    vecTime = [vecTime; output_curr.logData.time];
    vecState = [vecState; output_curr.logData.state];
    vecAmp = [vecAmp; output_curr.logData.amp];
    vecAliasedState = [vecAliasedState; output_curr.logData.aliasedState];

end

% obtain all metadata
LD0_adaptiveMetaData = ....
    vec_output{1}.adaptiveMetaData.LD0.(str_side(1));
LD1_adaptiveMetaData = ....
    vec_output{1}.adaptiveMetaData.LD1.(str_side(1));

%% now plot for RCS17

if strcmp(cfg.str_sub, 'RCS17')
    if any(strcmp(cfg.str_data_day, ...
            {'20230704', '20230706', '20230707', '20230711', '20230712', ...
            '20230713', '20230714', '20230718', '20230721', ...
            '20230722', '20230723', '20230726', '20230727', ...
            '20230729', '20230730'}))
        % unpack the log variables
        vecTime = vecTime;
        vecState = vecState;
        vecAmp = vecAmp;
        vecAliasedState = vecAliasedState;
        
        % unpack the PKG values
        if ~any(strcmp(cfg.str_data_day, cfg.str_no_pkg_data_day))
            timePKG = pkgWatchTable.Date_Time;
            PKGBradyData = pkgWatchTable.BK;
            PKGDyskData = pkgWatchTable.DK;
        else
            timePKG = NaN;
            PKGBradyData = NaN;
            PKGDyskData = NaN;
        end

       % sanity check
        if numel(cfg.vec_str_side) == 2
            error('Only provide one side')
        end

        if any(strcmp(cfg.vec_str_side, {'Left'}))
            color = [0, 0.4470, 0.7410]; smoothColor = [1, 0, 0];
            ylimState = [-0.5, 4.5]; ylimAliasedState = [-0.5, 1.5];
            ylimCurrent = [0.5, 4];
        elseif any(strcmp(cfg.vec_str_side, {'Right'}))
            color = [0, 0.4470, 0.7410]; smoothColor = [1, 0, 0];
            ylimState = [-0.5, 4.5]; ylimAliasedState = [-0.5, 1.5];
            ylimCurrent = [0.5, 4];
        end
    else
        error('Please change date signature above')
    end

else
    error("Analysis undefined for current subject")
end

%% Plotting

[figLogs, fFigure] = plotLogswPKG(vecTime, vecState, vecAmp, ...
    vecAliasedState, timePKG, PKGBradyData, ...
    PKGDyskData, LD0_adaptiveMetaData, str_side, cfg, ...
    'color', color, 'smoothColor', smoothColor, ...
    'ylimState', ylimState, 'ylimAliasedState', ylimAliasedState, ...
    'ylimCurrent', ylimCurrent);

% save the output
saveas(figLogs, fullfile(pFigure, sprintf('%s.png', fFigure)));
close(figLogs);
