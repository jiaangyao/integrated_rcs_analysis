% Maria's revisions on Cora's original main script
% maria.olaru@ucsf.edu


%% Batch_DataConversion
 clear
 
 files = dir('*.mat');
 
 % //Maria's modifications
 % move the renamed files into their own directory and cd into it
 study_filepath = strjoin([files(1).folder, "/batch_converted"], "");
 if ~exist(study_filepath, 'dir')
     mkdir(study_filepath);
 end 
 % //
 
 for i=1:length(files)
     
     EcogFileName = files(i).name;
     EcogFileName = strrep(EcogFileName,'.mat','');
     
     ConvertMat2EcogFileRCSSept19(EcogFileName)
     movefile('*raw_ecog.mat', study_filepath); % MO add
 end
 
%% analyze data


cd(study_filepath) %MO add
files = dir('*raw_ecog.mat');

 % //Maria's modifications
 %
 % make fig & analyzed directories
 analyzed_filepath = strjoin([files(1).folder, "/analyzed"], "");
 if ~exist(analyzed_filepath, 'dir')
     mkdir(analyzed_filepath);
 end 
 
  analyzed_fig_filepath = strjoin([files(1).folder, "/analyzed", "/fig"], ...
     "");
 if ~exist(analyzed_fig_filepath, 'dir')
     mkdir(analyzed_fig_filepath);
 end
 % //
 
for i =1 : length(files)
    % load the data and convert in mat.file
    file_name=files(i).name;
    load(file_name)
    EcogPreprocessing_ResamplingFirst(file_name)
    file_name =  strrep(file_name,'_raw','');
    file_name = [file_name(1:end-4) '_filt'];
    load(file_name)
    [ecog, lfp, bad] = ReReferenceData(ecog, lfp,2,file_name);
    
    % remove reference cross lead
    if ~isempty(strfind(file_name,'bi'))
        ecog.contact(4).signal_ref=[];
        lfp.contact(4).signal_ref=[];
    end
    save(name,'ecog','lfp','aux','bad','emg','name')
    
    
    if ~isempty(strfind(file_name,'SSEP'))
        SSEP_analysis_RCS(ecog,aux,file_name);
    elseif ~isempty(strfind(file_name,'mvt'))
        if i==2
        [smoothdata] = eegfilt(aux.chan(3).signal,1000,1,10);
        else
            [smoothdata] = eegfilt(aux.chan(2).signal,1000,1,10);
        end
        x = abs(hilbert(smoothdata));
        plot(x)
% plot(abs(emg.chan(1).signal-emg.chan(2).signal))
        % plot(aux.chan(1).signal)
        %IPAD_TRIG =input('ipad threshold');
        IPAD_TRIG = 2; %MO question: What is this variable?
        [trial_onset]=find_trial_onset(x,IPAD_TRIG);
        ETA_spectrogram_RCS(name,ecog,trial_onset)
        if ~isempty(strfind(file_name,'lfp'))
            ETA_spectrogram_RCS_lfp(name,lfp,trial_onset)
        end
    elseif ~isempty(strfind(file_name,'rest'))
        save(file_name,'bad','lfp','-append')
        %MultiContactPSD_RCS(ecog,bad,file_name);
         %MultiContactComodulogram_rest_RCS(ecog, file_name, 0)
        if ~isempty(strfind(file_name,'lfp'))
            MultiContactPSD_RCS_lfp(lfp,bad,file_name);
         MultiContactComodulogram_rest_RCS_lfp(lfp, file_name, 0)
        end
    end
    clear ecog aux emg
    close all
    
    % // MO additions
    if ~isempty(dir('*filt*.mat'))
        movefile('*filt*.mat', analyzed_filepath);
    end
    
    if ~isempty(dir('*ETA*.mat'))
        movefile('*ETA*.mat', analyzed_filepath); 
    end
    
    if ~isempty(dir('*.fig'))
        movefile('*.fig', analyzed_filepath);
    end
    % // 
end

