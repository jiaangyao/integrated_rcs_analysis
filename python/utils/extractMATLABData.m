% str_subject = "RCS02";
% str_subject = "RCS08";
str_subject = "RCS11";
% str_subject = "RCS12";
% str_subject = "RCS18";

bool_use_wArt = false;
if bool_use_wArt
    p_input = sprintf("/home/jyao/local/data/starrlab/Structured_aDBS_pipeline/Data/%s/Step3_in_clinic_neural_recordings/", str_subject);
    f_input = sprintf("%s_Step3_DataLab_paper_wArt", str_subject);
else
    p_input = sprintf("/home/jyao/local/data/starrlab/Structured_aDBS_pipeline/Data/%s/Step3_in_clinic_neural_recordings/old", str_subject);
    f_input = sprintf("%s_Step3_DataLab", str_subject);
end

% load the data
load(fullfile(p_input, f_input));
l_table = dataLab.Step3.L;
r_table = dataLab.Step3.R;

% save the output csv
f_output_left = sprintf("%s_L_table.csv", str_subject);
f_output_right = sprintf("%s_R_table.csv", str_subject);
writetable(l_table, fullfile(p_input, f_output_left));
writetable(r_table, fullfile(p_input, f_output_right));