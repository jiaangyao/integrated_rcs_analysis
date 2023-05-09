
function MultiContactComodulogram_rest_RCS(ecog, name, surrogate)

% 
% lfp=[];
% load(name)
name = strrep(name,'_ecog_filt.mat','');


num_chan = length(ecog.contact);

%% Define the Amplitude- and Phase- Frequencies

PhaseFreqVector=[4:2:50];
AmpFreqVector=[10:4:200];

PhaseFreq_BandWidth=2;
AmpFreq_BandWidth=4;


%% For comodulation calculation (only has to be calculated once)
nbin = 18;
position=zeros(1,nbin); % this variable will get the beginning (not the center) of each phase bin (in rads)
winsize = 2*pi/nbin;
for j=1:nbin
    position(j) = -pi+(j-1)*winsize;
end

for chan =1:num_chan
    %% Remove zeros added at the end of the ecog signal
        signal = ecog.contact(chan).signal_ref;
         Fs = ecog.Fs(chan);
    if length(signal)>=61000 % file must have the same duration 1 min
        signal=signal(:,1:60000);
    end
    data_length = length(signal);
    
    if ~isempty(signal)
        
        %% Do filtering and Hilbert transform on CPU
        
        'CPU filtering'
        tic
        AmpFreqTransformed = zeros(length(AmpFreqVector), data_length);
        PhaseFreqTransformed = zeros(length(PhaseFreqVector), data_length);
        
        for ii=1:length(AmpFreqVector)
            Af1 = AmpFreqVector(ii)-AmpFreq_BandWidth/2;
            Af2=AmpFreqVector(ii)+AmpFreq_BandWidth/2;
            AmpFreq=eegfilt_FIR(signal,Fs,Af1,Af2); % just filtering
            AmpFreqTransformed(ii, :) = abs(hilbert(AmpFreq)); % getting the amplitude envelope
        end
        
        for jj=1:length(PhaseFreqVector)
            Pf1 = PhaseFreqVector(jj) - PhaseFreq_BandWidth/2;
            Pf2 = PhaseFreqVector(jj) + PhaseFreq_BandWidth/2;
            PhaseFreq=eegfilt_FIR(signal,Fs,Pf1,Pf2); % this is just filtering
            PhaseFreqTransformed(jj, :) = angle(hilbert(PhaseFreq)); % this is getting the phase time series
        end
        % PhaseFreqTransformed = PhaseFreqTransformed(:,500: end-500);
        % AmpFreqTransformed = AmpFreqTransformed(:,500: end-500);
        toc
        
        % clear 'PhaseFreqTransformed' 'AmpFreqTransformed'
        %% Do comodulation calculation
        'Comodulation loop'
        
        counter1=0;
        for ii=1:length(PhaseFreqVector)
            counter1=counter1+1;
            ii
            counter2=0;
            for jj=1:length(AmpFreqVector)
                counter2=counter2+1;
                
                [MI,MeanAmp]=ModIndex_v2(PhaseFreqTransformed(ii, :), AmpFreqTransformed(jj, :), position);
                Comodulogram(counter1,counter2,chan)=MI;
                x = 10:20:360;
                [val,pos]=max(MeanAmp);
                Comodulogram_phase(counter1,counter2,chan) = x(pos);
                if surrogate ==1
                    if chan == M1_ch-1 || chan == M1_ch
                        numpoints=size(AmpFreqTransformed,2);
                        numsurrogate=200; %% number of surrogate values to compare to actual value
                        minskip=Fs; %% time lag must be at least this big
                        maxskip=numpoints-Fs; %% time lag must be smaller than this
                        skip=ceil(numpoints.*rand(numsurrogate*2,1));
                        skip((skip>maxskip))=[];
                        skip(skip<minskip)=[];
                        skip=skip(1:numsurrogate,1); % creates vector with of time lags "tau" (the skip values) used for surrogate MIs
                        surrogate_m=zeros(numsurrogate,1);
                        
                        for s=1:numsurrogate
                            Amp_surr =[AmpFreqTransformed(jj,skip(s):end) AmpFreqTransformed(jj,1:skip(s)-1)];
                            [MI_S,MeanAmp_S]=ModIndex_v2(PhaseFreqTransformed(ii,:), Amp_surr, position);
                            MI_surr(s) = MI_S;
                        end
                        
                        % fit gaussian to surrogate data, uses normfit.m from MATLAB Statistics toolbox
                        [surrogate_mean,surrogate_std]=normfit(MI_surr);
                        Mean_surr(counter1,counter2,chan)=surrogate_mean;
                        Std_surr(counter1,counter2,chan)=surrogate_std;
                        Comodulogram_surr(counter1,counter2,chan)=(abs(MI)-surrogate_mean)/surrogate_std;
                        p_surr(counter1,counter2,chan)=  prctile(MI_surr,99);
                        Comodulogram_surr(counter1,counter2,chan)=MI;
                        if MI<prctile(MI_surr,99)
                            Comodulogram_surr(counter1,counter2,chan)=0;
                        end
                        
                    else
                        Comodulogram_surr(counter1,counter2,chan)=MI;
                        Mean_surr(counter1,counter2,chan)=nan;
                        Std_surr(counter1,counter2,chan)=nan;
                        p_surr(counter1,counter2,chan)=nan;
                        %                     Comodulogram_surr(counter1,counter2,chan)=(abs(Comodulogram(counter1,counter2,chan))-Mean_surr(counter1,counter2,chan))./Std_surr(counter1,counter2,chan);
                    end
                else
                    Comodulogram_surr(counter1,counter2,chan)=MI;
                    Mean_surr(counter1,counter2,chan)=nan;
                    Std_surr(counter1,counter2,chan)=nan;
                    p_surr(counter1,counter2,chan)=nan;
                    %                     Comodulogram_surr(counter1,counter2,chan)=(abs(Comodulogram(counter1,counter2,chan))-Mean_surr(counter1,counter2,chan))./Std_surr(counter1,counter2,chan);
                end
            end
        end
        toc
    end
end
save([name '_Com'],'Comodulogram_surr','Comodulogram','Comodulogram_phase','p_surr','Mean_surr','Std_surr','PhaseFreqVector','PhaseFreq_BandWidth','AmpFreqVector','AmpFreq_BandWidth');


%% Graph comodulogram
Comodulogram_surr(1,1,:)=0.00001;
Clim2 = 0.0005;%max(max(max(Comodulogram(:,:,10))));
Clim1 = 0;%min(min(min(Comodulogram(:,:,10))));
figure;
for i = 1:size(Comodulogram_surr,3)
    subplot(2,4,i)
        
        C=squeeze(Comodulogram(:,:,i));
        contourf(PhaseFreqVector+PhaseFreq_BandWidth/2,AmpFreqVector+AmpFreq_BandWidth/2,C',30,'lines','none')
        set(gca,'fontsize',14)

        ylabel('Amplitude Frequency (Hz)')
        xlabel('Phase Frequency (Hz)')

        caxis([Clim1 Clim2])
        if i <4
            title(['L ' num2str(i+7) '-' num2str(i+8) ]); % allows title to have file name
        else
            title(['R ' num2str(i+3) '-' num2str(i+4) ]);
        end
end
    saveas(gcf,[name '_Com'],'png');

set(gcf,'units','normalized','outerposition',[0 0 1 1])
    saveas(gcf,[name '_Com'],'png');
close all

