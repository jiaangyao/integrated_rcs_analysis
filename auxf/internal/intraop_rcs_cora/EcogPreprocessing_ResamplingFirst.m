
function EcogPreprocessing_ResamplingFirst(EcogFileName)

% EcogFileName = [EcogFileName '_raw_ecog'];
load(EcogFileName)

%% resample data
if ~isempty(ecog)
for i = 1:length(ecog.contact)
    ecog.contact(i).signal= ecog.contact(i).raw_signal; 
    if ~isempty(ecog.contact(i).signal)
        if round(ecog.Fs(i))== 2750
            f1 = 2^10;
            f2 = (4^4)*11;
        elseif round(ecog.Fs(i))== 3052
            f1 = 2^10;
            f2 = 5^5;
        else
            f1 = 1000;
            f2 = ecog.Fs(i);
        end
        ecog.contact(i).signal=resample(ecog.contact(i).signal,f1,f2);
        ecog.Fs(i)=1000;
    end
end
end

if ~isempty(lfp)
    
    for i = 1:length(lfp.contact)
        if ~isempty(lfp.contact(i).raw_signal)
             lfp.contact(i).signal= lfp.contact(i).raw_signal;
            if round(lfp.Fs(i))== 2750
                f1 = 2^10;
                f2 = (4^4)*11;
            elseif round(lfp.Fs(i))== 3052
                f1 = 2^10;
                f2 = 5^5;
            else
                f1 = 1000;
                f2 = lfp.Fs(i);
            end
            
            lfp.contact(i).signal=resample(lfp.contact(i).signal,f1,f2);
            lfp.Fs(i)=1000;
        end
    end
end

if ~isempty(aux)
    for i = 1:length(aux.chan)
         aux.chan(i).signal= aux.chan(i).raw_signal;
%         if round(aux.Fs(i))== 2750
            f1 = 2^10;
            f2 = (4^4)*11;
%         elseif round(aux.Fs(i))== 24414
%             f1 = 2^10;
%             f2 = (5^5)*8;
%         else
%             f1 = 1000;
%             f2 = aux.Fs(i);
%         end
        
        aux.chan(i).signal=resample(aux.chan(i).signal,f1,f2);
        aux.Fs(i) = 1000;
    end
end
if ~isempty(emg)
    for i = 1:length(emg.chan)
        emg.chan(i).signal= emg.chan(i).raw_signal;
        if round(emg.Fs(i))== 2750
            f1 = 2^10;
            f2 = (4^4)*11;
        elseif round(emg.Fs(i))== 24414
            f1 = 2^10;
            f2 = (5^5)*8;
        else
            f1 = 1000;
            f2 = emg.Fs(i);
        end
        emg.chan(i).signal=resample(emg.chan(i).signal,f1,f2);
        emg.Fs(i)=1000;
    end
end

% %% remove DC offset and low freq
% for k=1:length(ecog.contact)
%     if ~isempty(ecog.contact(k).signal)
%         ecog.contact(k).signal= eegfilt(ecog.contact(k).signal,round(ecog.Fs(k)),1,[]); %notch out at 60
%         ecog.contact(k).signal= ecog.contact(k).signal-mean(ecog.contact(k).signal); %notch out at 60
%         ecog.contact(k).signal=[];
%     end
% end
% if ~isempty(lfp)
%     
%     for k=1: length(lfp.contact)
%         if ~isempty(lfp.contact(k).signal)
%             lfp.contact(k).signal= eegfilt(lfp.contact(k).signal,round(lfp.Fs(k)),1,[]); %notch out at 60
%             lfp.contact(k).signal= lfp.contact(k).signal-mean(lfp.contact(k).signal); %notch out at 60
%             lfp.contact(k).signal=[];
%         end
%     end
% end
% if ~isempty(aux)
%     for k=1:length(aux.chan)
%         aux.chan(k).signal = aux.chan(k).signal-mean(aux.chan(k).signal);
%         aux.chan(k).signal=[];
%     end
% end
% if ~isempty(emg)
%     for k=1:length(emg.chan)
%         emg.chan(k).signal = emg.chan(k).signal-mean(emg.chan(k).signal);
%         emg.chan(k).signal=[];
%     end
% end


%% save data
EcogFileName =  strrep(EcogFileName,'_raw','');
EcogFileName = [EcogFileName(1:end-4) '_filt'];
name = EcogFileName ;
save(EcogFileName,'ecog','lfp','aux','emg','name');

% %% find stim artifact
% if ~isempty(strfind(EcogFileName,'DBS')) || ~isempty(strfind(EcogFileName,'stim'))
%     WINDOW = 512;           % segment length and Hamming window length for welch's method
%     NOVERLAP = 256;         % # signal samples that are common to adjacent segments for welch's method
%     NFFT = 512;
%     epoch = 1000;
%     slid=100;
%     stim = [1: slid :length(ecog.contact(17).signal)-epoch] ;
%     [psd,f] = pwelch(ecog.contact(17).signal,WINDOW ,NOVERLAP,NFFT,Fs);
%     stim_peak = nan*ones(1,length(stim)-1);
%
%     freq = find(f>= 150 & f<= 200);
%     [v,p] = max(psd(freq));
%     f_max = p + freq(1)-1;
%     f_stim = f(f_max);
%
%     for t = 1: length(stim)-1
%         tt = stim(t);
%         [psd,f] = pwelch(ecog.contact(17).signal(tt:tt+epoch),WINDOW ,NOVERLAP,NFFT,Fs);
%         stim_peak(t) = psd(f_max);
%     end
%
%     figure
%     plot(stim_peak)
%     title(['f=  ' num2str(f_stim)])
%     saveas(gcf,[EcogFileName '_stim_art'],'fig');
%     % normalize signal
%     %     std_stim = std(stim_peak);
%     %     mean_stim = mean(stim_peak);
%     %     sup_mean = find(stim_peak > mean(stim_peak));
%     sup_mean = find(stim_peak > 1);
%
%     [pos,n] = evFindGroups(sup_mean,10,50);
%     if ~isempty(pos)
%     start_stim = sup_mean(pos(1));
%     stop_stim = sup_mean(pos(2));
%
%     [n1_b, n1_a]=butter(3,2*[f_stim-3 f_stim+3]/Fs,'stop');
%     for k=1:length(ecog.contact)
%         ecog.contact(k).signal=filtfilt(n1_b, n1_a, ecog.contact(k).signal); %notch stim art
%     end
%     save (EcogFileName,'start_stim','stop_stim','f_stim','-append')
% end

