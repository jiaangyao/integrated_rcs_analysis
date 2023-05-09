function ETA_spectrogram_RCS(name,lfp,trial_onset)

% use for preprocessed, downsampled data
% plots raw spectrograms

%% define variables
Fs=lfp.Fs(1);
WINDOW = 512*Fs/1000;           
NOVERLAP = 462*Fs/1000;                
NFFT = 512*Fs/1000;     

FRAME_ADVANCE=WINDOW-NOVERLAP;
PRE = 2;      % time (sec) before event
POST = 2;     % time (sec) after event

ADD = .5;     % add more time to increase # windows in each snippet
BL = [2 1];  % baseline period before movement onset for calculating percent power change


%% calculate normalized time-varying PSD 
% calculate spectrogram for all lfp/LFP
% gives the fft of each segment in each column, and the next column moves in time by window-noverlap samples
xx = find(trial_onset - PRE*Fs>1);
trial_onset  = trial_onset (xx);
xx = find(trial_onset +POST*Fs<length(lfp.contact(1).signal_ref));
trial_onset  = trial_onset (xx);

event_onset = trial_onset;
trials_ok =1:length(event_onset);

n_data_ch=length(lfp.contact)-1;
n_epochs = length(event_onset);



for i = 1:n_data_ch
    if ~isempty(lfp.contact(i).signal_ref)
    data=lfp.contact(i).signal_ref;
    for jj=1:length(trials_ok)-1; 
        j=trials_ok(jj);
        first = int32(event_onset(j)-(Fs*PRE)-WINDOW/2); % WINDOW/2 offset will make spectrogram start at moveonset-PRE at appropriately centered PSD
        last = int32(event_onset(j)+(Fs*POST)+WINDOW/2);
        snip = data(first:last);
        %calculate spectrogram of snippet 
        %3D matrix 'S' has the fft power value stored in [frequncy,time,trials] arrangement
        S(:,:,jj) = spectrogram(snip,WINDOW,NOVERLAP,NFFT,Fs); %#ok<AGROW>
    end

    %find the magnitude of the fft represented in each column of S
    S_mag=abs(S);

    %calculate average across all epochs 
    %note: S_mag contains spectrograms from each epoch in the 3rd dimension. The average across all epochs are then calculated and stored in the 
    %3rd dimension of S_mag_mean.  S_mag_mean collects averaged spectrogram for each data channel in the 3rd dimension.DO NOT GET CONFUSED!
    S_mag_mean(:,:,i) = nanmean(S_mag,3); %#ok<AGROW>
    S_mag_median(:,:,i) = nanmedian(S_mag,3); %#ok<AGROW>
    
    % clear some variables before next loop, this is probably not necessary but do it just in case
    clear data S S_mag;
    end
end

%% trial mean plot

%setup up the frequency (faxis)and time (taxis) axes data
[nfchans,nframes] = size(S_mag_mean(:,:,1));
nfchansteps = nfchans - 1;
maxfreq = Fs/2;
faxis = maxfreq*(0:nfchansteps)/nfchansteps;
t_res = FRAME_ADVANCE/Fs; % temporal resolution of spectrogram (sec)
taxis = (0:(nframes-1))* t_res;
taxis = taxis -PRE; %shift by PRE

% normalize to baseline values, mean plot
    A1norm = S_mag_mean;
    % to plot A with colors representing the log10 of power, uncomment this line:
    % A1plot = log10(S_mag_mean);
    first = int32(((PRE-BL(1))/t_res)+1);
    last = int32((BL(2))/t_res);
    % to plot A with colors representing raw data values, uncomment this line:
    A1plot = S_mag_mean;
    for i = 1:n_data_ch
        for j = 1:nfchans
            bl = A1norm(j,first:last,i);
            blmean = mean(bl);
            A1plot(j,:,i) = A1plot(j,:,i)/blmean; 
        end
    end

% plot TRIAL ONSET spectrogram for all lfp/lfp data 
hf1 = figure;
ff = find(faxis<=50);
val1 = min(min(min(A1plot(ff,:,:))));
val2 = max(max(max(A1plot(ff,:,:))));
clims1 = [val1 val2];
clims1 = [0.5 2];

for i = 1:n_data_ch%/2
    subplot(2,4,i);
    hold(gca,'on');
    % make the time-frequency plot
    tmp1 = A1plot(1:150,:,i); %chopping A1plot will allow the whole colobar to be represented
    faxis_new = faxis(1:150);
    imagesc(taxis,faxis_new,tmp1,clims1);
    if i <4
    title(['L ' num2str(i-1) '-' num2str(i) ]); % allows title to have file name
    else
        title(['R ' num2str(i-5) '-' num2str(i-4) ]);
    end
%     imagesc(taxis,faxis,A2plot(:,:,i),clims1);
    %plot vertical bar at movement onset
    plot([0 0],ylim,'k:');
    hold(gca,'off');
    % set the y-axis direction (YDir) to have zero at the bottom
    set(gca,'YDir','normal');
    % set xlim and ylim
    set(gca,'Xlim',[0-PRE POST]);
    set(gca,'Ylim',[0 150]);
   
     caxis([0.5 2])
end


% put a color scale indicator next to the time-frequency plot
% colorbar([0.9307 0.1048 0.02354 0.8226]);
%  colorbar([0.8 1.2]);
% save the figure
set(gcf,'units','normalized','outerposition',[0 0 1 1])
saveas(gcf,[name(1:end-5) '_ETA_lfp.fig']);
% saveas(gcf,[name(1:end-5) '_ETAmedian_type_all.fig']);

%% trial median plot

% normalize to baseline values, median plot

    %A2norm = log10(S_mag_median);
    A2norm = S_mag_median;

    % to plot A with colors representing the log10 of power, uncomment this line:
    %A2plot = log10(S_mag_median);
    A2plot=S_mag_median;
    first = int32(((PRE-BL(1))/t_res)+1);
    last = int32((BL(2))/t_res);
    % to plot A with colors representing raw data values, uncomment this line:
    A2plot = S_mag_median;
    for i = 1:n_data_ch
        for j = 1:nfchans
            bl = A2norm(j,first:last,i);
            blmean = mean(bl);
            A2plot(j,:,i) = A2plot(j,:,i)/blmean; 
        end
    end

    
% hf2 = figure;
% ff = find(faxis<=50);
% val1 = min(min(min(A2plot(ff,:,:))));
% val2 = max(max(max(A2plot(ff,:,:))));
% clims1 = [val1 val2];
% 
% for i = 1:n_data_ch%/2
%     subplot(2,4,i);
%     hold(gca,'on');
%     % make the time-frequency plot
%     tmp1 = A2plot(1:150,:,i); %chopping A2plot will allow the whole colobar to be represented
%     faxis_new = faxis(1:150);
%     imagesc(taxis,faxis_new,tmp1,clims1);
% %     imagesc(taxis,faxis,A2plot(:,:,i),clims1);
%     %plot vertical bar at movement onset
%     plot([0 0],ylim,'k:');
%     hold(gca,'off');
%     
%     % set the y-axis direction (YDir) to have zero at the bottom
%     set(gca,'YDir','normal');
%     % set xlim and ylim
%     set(gca,'Xlim',[0-PRE POST]);
%     set(gca,'Ylim',[0 150]);
% 
%     % axis labels/title
%     if i==1
%         title([name(1:end-5) '  aligned on image onset   mvt onset ' ]);
%             xlabel('time (sec)');
%             ylabel('frequency (Hz)');
%     end
%      caxis([0.5 2])
% end
% 
% % put a color scale indicator next to the time-frequency plot
% % colorbar([0.9307 0.1048 0.02354 0.8226]);
% %  colorbar([0.8 1.2]);
% % save the figure
% set(gcf,'units','normalized','outerposition',[0 0 1 1])
% % saveas(gcf,[name(1:end-5) '_ETAmean_trialtype_all.fig']);
% % save([name(1:end-5) '_ETA_trialtype_all.mat'], 'S_mag_median', 'S_mag_mean','WINDOW','NOVERLAP','NFFT','FRAME_ADVANCE','PRE','POST','ADD','BL');
% saveas(gcf,[name(1:end-5) '_ETAmean.fig']);
save([name(1:end-5) '_ETA_lfp.mat'], 'S_mag_median', 'S_mag_mean','WINDOW','NOVERLAP','NFFT','FRAME_ADVANCE','PRE','POST','ADD','BL');
end

