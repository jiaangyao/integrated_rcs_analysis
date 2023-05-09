function [t_wnan_out, data_wnan_out, idx_valid_epoch_out_curr] = correctIsolatdChunks(t_wnan, ...
    data_wnan, idx_start_curr, idx_valid_epoch_curr, pattern_curr)

t_wnan_out = t_wnan(:, :);
data_wnan_out = data_wnan(:, :);
idx_valid_epoch_out_curr = [];

half_span = idivide(numel(pattern_curr), int16(2));
i_start = 1 + half_span;
i_end = numel(idx_valid_epoch_curr) - half_span;
for i = i_start:i_end
    idx_valid_epoch_curr_test = idx_valid_epoch_curr(i-half_span: i+half_span);
    if all(idx_valid_epoch_curr_test == pattern_curr)
        idx_valid_epoch_out_curr = [idx_valid_epoch_out_curr, 0];
    else
        idx_valid_epoch_out_curr = [idx_valid_epoch_out_curr, ...
            idx_valid_epoch_curr(i)];
    end
end

% correct for epochs at start
bool_zeropad_start = idx_valid_epoch_out_curr(1) == 0;
if bool_zeropad_start
    % if first one is invalid epoch/
    idx_valid_epoch_out_curr = [zeros(1, half_span), idx_valid_epoch_out_curr];
else
    % if the first one is valid epoch
    for i = 1:half_span
        idx_valid_epoch_out_curr = [idx_valid_epoch_curr(i), ...
            idx_valid_epoch_out_curr];
    end
end

% now correct for epochs at end
bool_zeropad_end = idx_valid_epoch_out_curr(end) == 0;
len_curr = numel(idx_valid_epoch_out_curr);
if bool_zeropad_end
    % if last one is invalid epoch/
    idx_valid_epoch_out_curr = [idx_valid_epoch_out_curr, zeros(1, half_span)];
else
    % if the last one is valid epoch
    for i = 1:half_span
        idx_valid_epoch_out_curr = [idx_valid_epoch_out_curr, ...
            idx_valid_epoch_curr(i + len_curr)];
    end
end

% sanity check
if numel(idx_valid_epoch_out_curr) ~= numel(idx_valid_epoch_curr)
    error("These two sohuld be equal");
end

% now locate where changes occured;
idx_change_needed = idx_valid_epoch_curr - idx_valid_epoch_out_curr;
idx_rel_change = find(idx_change_needed == 1);
for i = 1:numel(idx_rel_change)
    % sanity check by computing chunks that are too isolated
    t_wnan_out(idx_start_curr(idx_rel_change(i)): idx_start_curr(idx_rel_change(i)+1)-1) = nan;
    data_wnan_out(idx_start_curr(idx_rel_change(i)): idx_start_curr(idx_rel_change(i)+1)-1) = nan;
end

end