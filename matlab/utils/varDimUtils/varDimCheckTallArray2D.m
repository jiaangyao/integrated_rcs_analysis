function boolTall = varDimCheckTallArray2D(inputArray)
%% varDimUtils script - checking array is tall
% Checks that an array (also tables and cells) is tall, having more
% entries in dim 1 than dim 2, 
% 
% INPUT:
% inputArray            - array/cell/table: input array to be queried


%% Check that array is tall
% create null output
boolTall = true;

% check that array is 2D
inputDim = size(inputArray);
assert(length(inputDim) == 2, 'Input array should be 2D')

% now check array is tall
assert(inputDim(1) >= inputDim(2), 'Array should be tall')


end