
% just top accuracy
acc = [];
auc = [];
for i = 1:length(output.Step3.medlev)
    acc = [acc, output.Step3.medlev{i}.caSing.R(1)];
    auc = [auc, output.Step3.medlev{i}.aucSing.R(1)];
end


% obtain the full accuracy
p_output = '/home/jyao/Downloads/biomarker_id/model_id';
f_output = 'RCS02_R_med_level_stats.mat';

acc = [];
auc = [];
auc_comb = [];
for i = 1:length(output.Step3.medlev)
    acc = [acc, output.Step3.medlev{i}.caSing.R];
    auc = [auc, output.Step3.medlev{i}.aucSing.R];
    auc_comb = [auc_comb, output.Step3.medlev{i}.aucSFS.R];
end

acc = acc'; auc = auc'; auc_comb= auc_comb';
save(fullfile(p_output, f_output), 'acc', 'auc', 'auc_comb');