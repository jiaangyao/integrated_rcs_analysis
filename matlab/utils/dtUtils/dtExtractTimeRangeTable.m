function output_data = extractTimeRangeData(t_abs_curr, data_curr, idx_chan, ...
    time_ranges_class1_in, time_ranges_class2_in)
    
    % change the timezone for some of the data
    time_ranges_class1 = time_ranges_class1_in;
    time_ranges_class2 = time_ranges_class2_in;

    time_ranges_class1.TimeZone = t_abs_curr.TimeZone;
    time_ranges_class2.TimeZone = t_abs_curr.TimeZone;
    
    % obtain data corresponding to class 1
    data_class1_full = [];
    for i_c1 = 1:size(time_ranges_class1, 1)
        idx_range_curr = t_abs_curr >= time_ranges_class1(i_c1, 1) & ...
            t_abs_curr < time_ranges_class1(i_c1, 2);
        data_range_curr = data_curr(idx_range_curr, idx_chan);

        data_class1_full = [data_class1_full; data_range_curr];
    end

    % obtain data corresponding to class 2
    data_class2_full = [];
    for i_c2 = 1:size(time_ranges_class2, 1)
        idx_range_curr = t_abs_curr >= time_ranges_class2(i_c2, 1) & ...
            t_abs_curr < time_ranges_class2(i_c2, 2);
        data_range_curr = data_curr(idx_range_curr, idx_chan);

        data_class2_full = [data_class2_full; data_range_curr];
    end

    % sanity check
    if (size(data_class1_full, 1) + size(data_class2_full, 1)) ~= size(data_curr, 1)
        error("These two should be equal");
    end

    output_data = {data_class1_full, data_class2_full};

end