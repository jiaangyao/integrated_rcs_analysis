% str_subject = "RCS02";
% str_subject = "RCS08";
% str_subject = "RCS11";
% str_subject = "RCS12";
str_subject = "RCS18";

if strcmp(str_subject, "RCS02")
    str_side = 'R';
elseif strcmp(str_subject, "RCS08")
    str_side = 'R';
elseif strcmp(str_subject, "RCS11")
    str_side = 'L';
elseif strcmp(str_subject, "RCS12")
    str_side = 'L';
elseif strcmp(str_subject, "RCS18")
    str_side = 'L';
end


% obtain the path
p_input = sprintf("/home/jyao/local/data/starrlab/Structured_aDBS_pipeline/Data/%s/Step3_in_clinic_neural_recordings/", str_subject);
f_input = sprintf("%s_Step3_output_paper_wArt", str_subject);

% load the data
load(fullfile(p_input, f_input));


% obtain the full accuracy
p_output = '/home/jyao/Downloads/';
f_output = sprintf('%s_R_med_level_stats.mat', str_subject);


auc_sin = output.Step3.medlev.aucSing.(str_side);
auc_sfs = output.Step3.medlev.aucSFS.(str_side);
pb_sin = output.Step3.medlev.pbSing.(str_side);
pb_sfs = output.Step3.medlev.pbSFS.(str_side);

% auc_sin = output.Step3.medlev.aucSing;
% auc_sfs = output.Step3.medlev.aucSFS;
% pb_sin = output.Step3.medlev.pbSing;
% pb_sfs = output.Step3.medlev.pbSFS;

save(fullfile(p_output, f_output), 'auc_sin', 'auc_sfs', 'pb_sin', 'pb_sfs');