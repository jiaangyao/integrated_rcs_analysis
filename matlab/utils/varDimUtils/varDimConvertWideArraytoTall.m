function outputArray = varDimConvertWideArraytoTall(inputArray)
%% varDimUtils script - checking array shape
% Convert inputArray to tall array if it is wide
% 
% INPUT:
% inputArray            - array/cell/table: input array to be queried


%% Convert wide array to tall array

% check that array is 2D
assert(length(size(inputArray)) == 2, 'Input array should be 2D')

% return a transposed deepcopy if input is a wide array
if size(inputArray, 1) < size(inputArray, 2)
    outputArray = inputArray(:, :);
    outputArray = outputArray.';
end


end