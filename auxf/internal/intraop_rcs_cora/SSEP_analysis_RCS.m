function  SSEP_analysis_RCS(ecog,aux,file_name);
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
% signal=emg.chan(1).raw_signal;
plot(abs(signal))

%STIM_TRIG_ON = 150; %MO EDIT
STIM_TRIG_ON =input('stim threshold');

hold on
[v,inds] = findpeaks(abs(signal(15:end-15)),'threshold',STIM_TRIG_ON);
%inds = find(abs(signal(15:end-15))>=STIM_TRIG_ON);
% [pos,n] = evFindGroups(inds,400,1);
% stim_trig_time = inds(pos(1,:))+15;
stim_trig_time = inds+15;
plot(stim_trig_time,STIM_TRIG_ON,'*k')

% find and eliminate any stim trig times that might exceed the length of
% ecog data
% tmax = length(ecog.contact(1).signal_ref)-PST_STIM*Fs;
% stim_trig_time = stim_trig_time(stim_trig_time<tmax);
xx = find(stim_trig_time+PRE_STIM>1);
stim_trig_time = stim_trig_time(xx);
xx = find(stim_trig_time+POST_STIM<length(ecog.contact(1).signal_ref));
stim_trig_time = stim_trig_time(xx);

num_stim = length(stim_trig_time);
t = PRE_STIM:POST_STIM; % multiplied by 1000 to change to msec scale

% initialize
% diff_stim_epoch contains 5 layers of 2-D array corresponding to the 5
% montaged ecog pairs. The rows of each 2-D array contain a vector of raw
% ecog data around the time of each stimulation.  
diff_stim_epoch = zeros(num_stim,length(t),length(ecog.contact));

for i = 1:length(ecog.contact)-1
    if i~=4
    % parse each stim epoch from each contact pair
    for j = 1:num_stim
        tmp1 = stim_trig_time(j)+PRE_STIM;  % int32 used to keep index in integer format
        tmp2 = stim_trig_time(j)+POST_STIM;
        diff_stim_epoch(j,:,i) = ecog.contact(i).signal_ref(tmp1:tmp2);
    end
    end
end


mean_diff_stim_epoch = mean(diff_stim_epoch);

figure;
hold on
colors ={'r','g','b'}
for i = 1:3
    SSEPmin = min(mean_diff_stim_epoch(1,:,i)); % mix value of average
    SSEPmax = max(mean_diff_stim_epoch(1,:,i)); % max value of average
    C = i; % constant added to stack the waves for comparison such that the first contact pair is plotted at the top
%     z = (mean_diff_stim_epoch(1,:,i)-SSEPmin)/(SSEPmax-SSEPmin) + C;
    z = -(mean_diff_stim_epoch(1,:,i)-SSEPmin)/(SSEPmax-SSEPmin) + C; % SSEPs recorded using AO shows N20 as an up-going potential.  invert this with a negative sign to make it down-going.
    plot(t,z,colors{i}); 
end
xlabel('Time (msec)');
ylabel('normalized SSEPs');
ylm = ylim;
plot([0 0],[ylm(1) ylm(2)],'k--')
title(file_name);
legend('8-9','9-10','10-11')
hold off
saveas(gcf,[file_name 'SSEPL'])

figure;
hold on
colors ={'r','g','b'}
for i = 5:7
    SSEPmin = min(mean_diff_stim_epoch(1,:,i)); % mix value of average
    SSEPmax = max(mean_diff_stim_epoch(1,:,i)); % max value of average
    C = i; % constant added to stack the waves for comparison such that the first contact pair is plotted at the top
%     z = (mean_diff_stim_epoch(1,:,i)-SSEPmin)/(SSEPmax-SSEPmin) + C;
    z = -(mean_diff_stim_epoch(1,:,i)-SSEPmin)/(SSEPmax-SSEPmin) + C; % SSEPs recorded using AO shows N20 as an up-going potential.  invert this with a negative sign to make it down-going.
    plot(t,z,colors{i-4}); 
end
xlabel('Time (msec)');
ylabel('normalized SSEPs');
ylm = ylim;
plot([0 0],[ylm(1) ylm(2)],'k--')
title(file_name);
legend('8-9','9-10','10-11')
hold off
saveas(gcf,[file_name 'SSEPR'])


