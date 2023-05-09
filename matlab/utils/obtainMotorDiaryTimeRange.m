function [time_ranges_class1, time_ranges_class2] = ...
    obtainMotorDiaryTimeRange(md_reduced_curr, threshold_curr, str_md_feature, ...
        md_range, sess_start, sess_end)

    % first obtain the time points that are in between session time
    % note that both bounds should be inclusive
    bool_time_valid = table2array(md_reduced_curr(:, 'time')) >= sess_start & ...
        table2array(md_reduced_curr(:, 'time')) <= sess_end;
    md_reduced_valid_time = md_reduced_curr(bool_time_valid, :);
    
    % now obtain all valid time ranges of class 1 (1-indexed)
    time_ranges_class1 = [];
    for i = 1:(size(md_reduced_valid_time, 1) - 1)
        feature_curr = table2array(md_reduced_valid_time(i, str_md_feature));
        time_start_curr = table2array(md_reduced_valid_time(i, "time"));
        time_end_curr = table2array(md_reduced_valid_time(i + 1, 'time'));

        % append to cell above if valid
        if feature_curr < threshold_curr && feature_curr >= md_range(1)
            % if empty
            if size(time_ranges_class1, 1) == 0
                time_ranges_class1 = [time_start_curr, time_end_curr];
       
            else
            % otherwise append to existing cell
                time_temp = [time_start_curr, time_end_curr];
                time_ranges_class1 = [time_ranges_class1; time_temp];
            end
        end
    end

    % now obtain all valid time ranges for class 2
    time_ranges_class2 = [];
    for i = 1:(size(md_reduced_valid_time, 1) - 1)
        feature_curr = table2array(md_reduced_valid_time(i, str_md_feature));
        time_start_curr = table2array(md_reduced_valid_time(i, "time"));
        time_end_curr = table2array(md_reduced_valid_time(i + 1, 'time'));

        % append to cell above if valid
        if feature_curr >= threshold_curr && feature_curr <= md_range(2)
            % if empty
            if size(time_ranges_class2, 1) == 0
                time_ranges_class2 = [time_start_curr, time_end_curr];
       
            else
            % otherwise append to existing cell
                time_temp = [time_start_curr, time_end_curr];
                time_ranges_class2 = [time_ranges_class2; time_temp];
            end
        end
    end

    % sanity check of data length
    if (size(time_ranges_class1, 1) + size(time_ranges_class2, 1)) ~= ...
            (size(md_reduced_valid_time, 1) - 1)
        error('These two should be equal')
    end
end