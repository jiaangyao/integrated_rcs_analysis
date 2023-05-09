
function MultiContactPSD_RCS_lfp(lfp,bad,name)

% lfp=[];
psd=[];
psd_all=[];
% load(name)
name = strrep(name,'_ecog_filt.mat','');

num_chan = length(lfp.contact);

for i = 1:num_chan
    psd=[];
        if ~isempty( lfp.contact(i).signal_ref)
            [psd,F] = pwelch(lfp.contact(i).signal_ref,2^(nextpow2(lfp.Fs(i))),2^(nextpow2(lfp.Fs(i)/2)),2^(nextpow2(lfp.Fs(i))),lfp.Fs(i));
        end
    if ~isempty(psd)
    psd_all(:,i)=psd;
    t = find(F>=1 & F<=4);
    psd_delta(i) = nanmean(log10(psd(t)));
    t = find(F>=5 & F<=7);
    psd_theta(i) = nanmean(log10(psd(t)));
    t = find(F>=8 & F<=12);
    psd_alpha(i) = nanmean(log10(psd(t)));
    t = find(F>=13 & F<=30);
    psd_beta(i) = nanmean(log10(psd(t)));
    t = find(F>=13 & F<=20);
    psd_Lbeta(i) = nanmean(log10(psd(t)));
    t = find(F>=20 & F<=30);
    psd_Hbeta(i) = nanmean(log10(psd(t)));
    t = find(F>=50 & F<=150);
    psd_gamma(i) = nanmean(log10(psd(t)));
    
    norm_idx=find(F>=5 & F<=100); % use norm_idx to normalize by max power between 8-100Hz, SAS 11/24/09
    psd_norm=psd/mean(psd(norm_idx(1):norm_idx(end))); % normalize each column to its max value
    psd_norm_all(:,i)=psd_norm;
    t = find(F>=1 & F<=4);
    psd_norm_delta(i) = nanmean(log10(psd_norm(t)));
    t = find(F>=5 & F<=7);
    psd_norm_theta(i) = nanmean(log10(psd_norm(t)));
    t = find(F>=8 & F<=12);
    psd_norm_alpha(i) = nanmean(log10(psd_norm(t)));
    t = find(F>=13 & F<=30);
    psd_norm_beta(i) = nanmean(log10(psd_norm(t)));
    t = find(F>=13 & F<=20);
    psd_norm_Lbeta(i) = nanmean(log10(psd_norm(t)));
    t = find(F>=20 & F<=30);
    psd_norm_Hbeta(i) = nanmean(log10(psd_norm(t)));
    t = find(F>=30 & F<=50);
    psd_norm_gamma(i) = nanmean(log10(psd_norm(t)));
    end
end

save([name '_psd_lfp'],'psd_theta', 'psd_delta','psd_alpha','psd_beta','psd_Lbeta','psd_Hbeta','psd_gamma','psd_all','F'...
    ,'psd_norm_theta', 'psd_norm_delta','psd_norm_alpha','psd_norm_beta','psd_norm_Lbeta','psd_norm_Hbeta','psd_norm_gamma','psd_norm_all','bad');

%% plot data

% figure plot log PSD
logpsdall = log10(psd_all);
figure
for i = 1:size(psd_all,2)
    
    subplot(2,4,i)
    
    ha = gca;
    hold(ha,'on');
    plot(F,logpsdall(:,i),'k','LineWidth',2);
    if i<4
    title(['L ' num2str(i-1) '-' num2str(i) ]); 
    else
    title(['R ' num2str(i-5) '-' num2str(i-4) ]); 
    end        
    xlabel('frequency (Hz)');
    ylabel('log PSD');
    xlim([0 150]);
    ylim([-3 3]);
    hold on
    fill([12 30 30 12],[-3 -3 3 3],'g','EdgeColor','none','FaceAlpha',0.3);
    fill([70 150 150 70],[-3 -3 3 3],'y','EdgeColor','none','FaceAlpha',0.3)
end
saveas(gcf,[name '_logpsd_lfp'],'png');
set(gcf,'units','normalized','outerposition',[0 0 1 1])
saveas(gcf,[name '_logpsd_lfp'],'png');

