function out = allclose(a, b, varargin)
%% input parsing
p = inputParser;
p.KeepUnmatched = true;

% optional commands for handling the sampling rate parsing
addParameter(p, 'atol', 1e-5, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));
addParameter(p, 'rtol', 1e-8, ...
    @(x) validateattributes(x, {'double'}, {'nonempty'}));

parse(p,varargin{:});

% Handles incorrect inputs
UnmatchedParam = fieldnames(p.Unmatched);
if ~isempty(UnmatchedParam)
    error(['"',UnmatchedParam{1},'" is not a valid parameter.']);
end

% unpacking variable
atol = p.Results.atol;
rtol = p.Results.rtol;


%% perform equality check

out = isequal(size(a), size(b)) & all( abs(a(:)-b(:)) <= atol+rtol*abs(b(:)) );


end