

function ConvertMat2EcogFileRCS(EcogFileName)

% define structures
lfp_module=4;
if ~isempty(strfind(EcogFileName,'bi'))
    lfp_num=8;  % 1-4 L side 5-8 Rside
else
    lfp_num=4; % unilateral case
end
lfp_first=1;

emg_module=3;
emg_num=4;
emg_first=1;

aux_module=0; 
aux_num=4;
aux_first=1;

if ~isempty(strfind(EcogFileName,'bi'))
    ecog_num=8;  % 1-4 L side 5-8 Rside
else
    ecog_num=4; % unilateral case
end
raw2_rec_on_module2=1; % raw 1 and 2 recorded on different modules
% raw2_rec_on_module2=0; % raw 1 and 2 recorded on different modules

raw1_module = 1;
raw2_module = 2;


load(EcogFileName)
EcogFileName = strrep(EcogFileName,'.mat','');
name= [EcogFileName,'_raw_ecog'];

%% ecog/LFP

ecog = [];
module = [ones(16,1);2*ones(16,1);3*ones(16,1);4*ones(16,1)];
chan = [1:16 1:16 1:16 1:16]';
if raw2_rec_on_module2==1
    for i = 1 : ecog_num
        if i <15 % first raw
            if i<10 
                try
                    eval(['ecog.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(raw1_module) '___0' num2str(i) '___Array_1___0' num2str(i) ')']);
                    eval(['ecog.Fs(' num2str(i) ') =  CECOG_' num2str(raw1_module) '___0' num2str(i) '___Array_1___0' num2str(i) '_KHz*1000']);
                end
            else
                try
                    eval(['ecog.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(raw1_module) '___' num2str(i) '___Array_1___' num2str(i) ')']);
                    eval(['ecog.Fs(' num2str(i) ') =  CECOG_' num2str(raw1_module) '___' num2str(i) '___Array_1___' num2str(i) '_KHz*1000']);
                end
            end
        else % 2nd raw
            if i<24 
                try
                    eval(['ecog.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(raw2_module) '___0' num2str(i-14) '___Array_1___' num2str(i+2) ')']);
                    eval(['ecog.Fs(' num2str(i) ') =  CECOG_' num2str(raw2_module) '___0' num2str(i-14) '___Array_1___' num2str(i+2) '_KHz*1000']);
                end
            else 
                try
                    eval(['ecog.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(raw2_module) '___' num2str(i-14) '___Array_1___' num2str(i+2) ')']);
                    eval(['ecog.Fs(' num2str(i) ') =  CECOG_' num2str(raw2_module) '___' num2str(i-14) '___Array_1___' num2str(i+2) '_KHz*1000']);
                end
            end
        end
    end
else
    for i=1 : ecog_num
        if chan(i) <10 & i<10
            try
                eval(['ecog.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(module(i)) '___0' num2str(chan(i)) '___Array_1___0' num2str(i) ')']);
                eval(['ecog.Fs(' num2str(i) ') =  CECOG_' num2str(module(i)) '___0' num2str(chan(i)) '___Array_1___0' num2str(i) '_KHz*1000']);
            end
        elseif chan(i) <10 & i>10
            try
                eval(['ecog.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(module(i)) '___0' num2str(chan(i)) '___Array_1___' num2str(i) ')']);
                eval(['ecog.Fs(' num2str(i) ') =  CECOG_' num2str(module(i)) '___0' num2str(chan(i)) '___Array_1___' num2str(i) '_KHz*1000']);
            end
        else
            try
                eval(['ecog.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(module(i)) '___' num2str(chan(i)) '___Array_1___' num2str(i) ')']);
                eval(['ecog.Fs(' num2str(i) ') =  CECOG_' num2str(module(i)) '___' num2str(chan(i)) '___Array_1___' num2str(i) '_KHz*1000']);
            end
        end
    end
end

lfp = [];
if ~isempty(strfind(EcogFileName,'lfp'))
    for i=1:lfp_num
        if lfp_module==2
                eval(['lfp.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(lfp_module) '___0' num2str(lfp_first+i-1) '___Array_' num2str(lfp_module-1) '___' num2str(lfp_first+i-1+16) ')']); % some case recorded with module 3 instead of 4
                eval(['lfp.Fs(' num2str(i) ') =  CECOG_' num2str(lfp_module) '___0' num2str(lfp_first+i-1) '___Array_' num2str(lfp_module-1) '___' num2str(lfp_first+i-1+16) '_KHz*1000']);
        else
            if i<10
                eval(['lfp.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(lfp_module) '___0' num2str(lfp_first+i-1) '___Array_' num2str(lfp_module-1) '___0' num2str(lfp_first+i-1) ')']); % some case recorded with module 3 instead of 4
                eval(['lfp.Fs(' num2str(i) ') =  CECOG_' num2str(lfp_module) '___0' num2str(lfp_first+i-1) '___Array_' num2str(lfp_module-1) '___0' num2str(lfp_first+i-1) '_KHz*1000']);
            else
                eval(['lfp.contact(' num2str(i) ').raw_signal =  double(CECOG_' num2str(lfp_module) '___' num2str(lfp_first+i-1) '___Array_' num2str(lfp_module-1) '___' num2str(lfp_first+i-1) ')']); % some case recorded with module 3 instead of 4
                eval(['lfp.Fs(' num2str(i) ') =  CECOG_' num2str(lfp_module) '___' num2str(lfp_first+i-1) '___Array_' num2str(lfp_module-1) '___' num2str(lfp_first+i-1) '_KHz*1000']);
            end
        end
    end
end


emg=[];
if emg_num>0
    for i=emg_first:emg_first+emg_num-1
        if i<10
            eval(['emg.chan(' num2str(i) ').raw_signal =  double(CECOG_' num2str(emg_module) '___0' num2str(i) '___Array_' num2str(emg_module-1) '___0' num2str(i) ')']); % some case recorded with module 3 instead of 4
            eval(['emg.Fs(' num2str(i) ') =  CECOG_' num2str(emg_module) '___0' num2str(i) '___Array_' num2str(emg_module-1) '___0' num2str(i) '_KHz*1000']);
        else
            eval(['emg.chan(' num2str(i-emg_first+1) ').raw_signal =  double(CECOG_' num2str(emg_module) '___' num2str(i) '___Array_' num2str(emg_module-1) '___' num2str(i) ')']); % some case recorded with module 3 instead of 4
            eval(['emg.Fs(' num2str(i-emg_first+1) ') =  CECOG_' num2str(emg_module) '___' num2str(i) '___Array_' num2str(emg_module-1) '___' num2str(i) '_KHz*1000']);
        end
    end
end

aux=[];
if aux_num>0
    for i=aux_first:aux_first+aux_num-1
        if aux_module~=0
            if i<10
                eval(['aux.chan(' num2str(i) ').raw_signal =  double(CECOG_' num2str(aux_module) '___0' num2str(i) '___Array_' num2str(aux_module-1) '___0' num2str(i) ')']); % some case recorded with module 3 instead of 4
                eval(['aux.Fs(' num2str(i) ') =  CECOG_' num2str(aux_module) '___0' num2str(i) '___Array_' num2str(aux_module-1) '___0' num2str(i) '_KHz*1000']);
            else
                eval(['aux.chan(' num2str(i) ').raw_signal =  double(CECOG_' num2str(aux_module) '___' num2str(i) '___Array_' num2str(aux_module-1) '___' num2str(i) ')']); % some case recorded with module 3 instead of 4
                eval(['aux.Fs(' num2str(i) ') =  CECOG_' num2str(aux_module) '___' num2str(i) '___Array_' num2str(aux_module-1) '___' num2str(i) '_KHz*1000']);
            end
        else
                eval(['aux.chan(' num2str(i) ').raw_signal =  double(CANALOG_IN_' num2str(i) ')']); % some case recorded with module 3 instead of 4
                eval(['aux.Fs(' num2str(i) ') =  CANALOG_IN_' num2str(i)  '_KHz*1000']); % some case recorded with module 3 instead of 4
        end
    end
end


save(name ,'ecog','lfp','aux','emg','name')
