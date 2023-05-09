function boolExist = varDimCheckValidEntryExist(inputArray)
%% varDimUtils script - checking array has at least one row
% Checks that an array (also tables and cells) has at least one entry in
% dim 1
% 
% INPUT:
% inputArray            - array/cell/table: input array to be queried


%% Check that at least one row exists

% create null output and assert at least one row exists
boolExist = true;
assert(size(inputArray, 1) > 0, 'Array is empty')


end