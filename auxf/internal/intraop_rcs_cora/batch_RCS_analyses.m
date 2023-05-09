
%%% Batch_DataConversion
clear
% 
 files = dir('*.mat');
 for i=1:length(files)
     
     EcogFileName = files(i).name;
     EcogFileName = strrep(EcogFileName,'.mat','');
     
     ConvertMat2EcogFileRCSSept19(EcogFileName)
 end

%% analyze data
files = dir('*raw_ecog.mat');
for i =1 : length(files)
    % load the data and convert in mat.file
    file_name=files(i).name;
    load(file_name)
    EcogPreprocessing_ResamplingFirst(file_name) %outputs "_filt"
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
        if ~isempty(strfind(file_name,'Lmvt')) %MO: check that accel is plugged accordingly.. typically L = 2, R = 3
            [smoothdata] = eegfilt(aux.chan(2).signal,1000,1,10);
        else
            [smoothdata] = eegfilt(aux.chan(2).signal,1000,1,10); %MO change 3 -> 2 b/c convention diffs
             %[smoothdata] = eegfilt(emg.chan(5).signal,1000,1,10 - eegfilt(emg.chan(6).signal,1000,1,10)); %if eeg doesn't work, check emg
        end
        x = abs(hilbert(smoothdata));
        plot(x)
        %plot(abs(emg.chan(1).signal-emg.chan(2).signal)) %for debugging no
        %movement
        IPAD_TRIG =input('ipad threshold');
        [trial_onset]=find_trial_onset(x,IPAD_TRIG);
        ETA_spectrogram_RCS(name,ecog,trial_onset)
        if ~isempty(strfind(file_name,'lfp'))
            ETA_spectrogram_RCS_lfp(name,lfp,trial_onset)
        end
    elseif ~isempty(strfind(file_name,'rest'))
        save(file_name,'bad','lfp','-append')
        MultiContactPSD_RCS(ecog,bad,file_name);
         MultiContactComodulogram_rest_RCS(ecog, file_name, 0)
        if ~isempty(strfind(file_name,'lfp'))
            MultiContactPSD_RCS_lfp(lfp,bad,file_name);
         MultiContactComodulogram_rest_RCS_lfp(lfp, file_name, 0)
        end
    end
    %plot(aux.chan(3).signal) %MO debugging
    clear ecog aux emg
    close all
    
end
