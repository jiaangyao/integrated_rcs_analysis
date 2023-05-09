function  SSEP_analysis(ecog,aux,file_name);
% this m-file performs SSEP analysis on mat file generated from map format
% using Alpha Omega's mapfile converter program

%% Define variables
gain=1;
Fs= ecog.Fs(1); % sampling freq after downsampling

PRE_STIM = -0.05 * Fs;     % pre-stim period in sec
POST_STIM = 0.05*Fs;     % post-stim period in sec



%% Process stim trigger voltage channel

% trig_chan = aux.chan(3).raw; 
T=1/Fs;% the sampling rate in line above is multiplied by 1000 for KHz->Hz conversion
signal=aux.chan(1).signal;%XX origna 
plot(abs(signal))
STIM_TRIG_ON =input('stim threshold');
hold on
[val inds] = findpeaks(abs(signal),'Threshold',STIM_TRIG_ON);
stim_trig_time = inds;
plot(stim_trig_time,STIM_TRIG_ON,'*r')


num_stim = length(stim_trig_time);
t = PRE_STIM:POST_STIM; % multiplied by 1000 to change to msec scale

% initialize
% diff_stim_epoch contains 5 layers of 2-D array corresponding to the 5
% montaged ecog pairs. The rows of each 2-D array contain a vector of raw
% ecog data around the time of each stimulation.  
diff_stim_epoch = zeros(num_stim,length(t),length(ecog.contact));

for i = 1:length(ecog.contact)-1
    
    % parse each stim epoch from each contact pair
    for j = 1:num_stim
        tmp1 = stim_trig_time(j)+PRE_STIM;  % int32 used to keep index in integer format
        tmp2 = stim_trig_time(j)+POST_STIM;
        diff_stim_epoch(j,:,i) = ecog.contact(i).signal_ref(tmp1:tmp2);
    end
end


mean_diff_stim_epoch = mean(diff_stim_epoch);


figure;
if length(ecog.contact)>14
    
    subplot(1,2,1)
    hold on
    for i = 1:length(ecog.contact)/2-1
        SSEPmin = min(mean_diff_stim_epoch(1,:,i)); % mix value of average
        SSEPmax = max(mean_diff_stim_epoch(1,:,i)); % max value of average
        C = i/2; % constant added to stack the waves for comparison such that the first contact pair is plotted at the top
        z = -(mean_diff_stim_epoch(1,:,i)-SSEPmin)/(SSEPmax-SSEPmin) + C; % SSEPs recorded using AO shows N20 as an up-going potential.  invert this with a negative sign to make it down-going.
        
        plot(t,z);
    end
    xlabel('Time (msec)');
    ylabel('normalized SSEPs');
    ylm = ylim;
    plot([0 0],[ylm(1) ylm(2)],'k--')
    title(file_name);
    xlim([-5 30])
    subplot(1,2,2)
    hold on
    for i = length(ecog.contact)/2: length(ecog.contact)-1
        SSEPmin = min(mean_diff_stim_epoch(1,:,i)); % mix value of average
        SSEPmax = max(mean_diff_stim_epoch(1,:,i)); % max value of average
        C = i/2; % constant added to stack the waves for comparison such that the first contact pair is plotted at the top
        z = -(mean_diff_stim_epoch(1,:,i)-SSEPmin)/(SSEPmax-SSEPmin) + C; % SSEPs recorded using AO shows N20 as an up-going potential.  invert this with a negative sign to make it down-going.
        plot(t,z);
    end
    xlabel('Time (msec)');
    ylabel('normalized SSEPs');
    ylm = ylim;
    xlim([-5 30])
    plot([0 0],[ylm(1) ylm(2)],'k--')
    title(file_name);
    
else
    hold on
    for i = 1:length(ecog.contact)/2-1
        SSEPmin = min(mean_diff_stim_epoch(1,:,i)); % mix value of average
        SSEPmax = max(mean_diff_stim_epoch(1,:,i)); % max value of average
        C = i/2; % constant added to stack the waves for comparison such that the first contact pair is plotted at the top
        z = -(mean_diff_stim_epoch(1,:,i)-SSEPmin)/(SSEPmax-SSEPmin) + C; % SSEPs recorded using AO shows N20 as an up-going potential.  invert this with a negative sign to make it down-going.
        
        plot(t,z);
    end
    xlabel('Time (msec)');
    ylabel('normalized SSEPs');
    ylm = ylim;
    plot([0 0],[ylm(1) ylm(2)],'k--')
    title(file_name);
    xlim([-5 30])
end

hold off

saveas(gcf,[file_name(1:end-4) '_SSEP'])

