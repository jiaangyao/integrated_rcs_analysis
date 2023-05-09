
function MultiContactPSD(ecog,lfp,bad,name)

% lfp=[];
psd=[];
% load(name)
name = strrep(name,'_ecog_filt.mat','');
if ~isempty(lfp)
    num_chan = length(ecog.contact) +length(lfp.contact);
else
    num_chan = length(ecog.contact);
end
num_chan = length(ecog.contact);
for i = 1:num_chan-1
    if i <=length(ecog.contact)
        if ~isempty( ecog.contact(i).signal)
            [psd,F] = pwelch(ecog.contact(i).signal_ref,2^(nextpow2(ecog.Fs(i))),2^(nextpow2(ecog.Fs(i)/2)),2^(nextpow2(ecog.Fs(i))),ecog.Fs(i));
        end
    else
        j = i-length(ecog.contact);
        if ~isempty( lfp.contact(j).signal)
            [psd,F] = pwelch(lfp.contact(j).signal_ref,2^(nextpow2(lfp.Fs(j))),2^(nextpow2(lfp.Fs(j)/2)),2^(nextpow2(lfp.Fs(j))),lfp.Fs(j));
        end
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

save([name '_psd'],'psd_theta', 'psd_delta','psd_alpha','psd_beta','psd_Lbeta','psd_Hbeta','psd_gamma','psd_all','F'...
    ,'psd_norm_theta', 'psd_norm_delta','psd_norm_alpha','psd_norm_beta','psd_norm_Lbeta','psd_norm_Hbeta','psd_norm_gamma','psd_norm_all','bad');

%% plot data

% figure plot log PSD
logpsdall = log10(psd_all);
figure
for i = 1:size(psd_all,2)
    
    subplot(3,14,i)
    
    ha = gca;
    hold(ha,'on');
    plot(F,logpsdall(:,i),'k','LineWidth',2);
    if i==1;
        title(['C' num2str(i) ]); % allows title to have file name
        xlabel('frequency (Hz)');
        ylabel('log PSD');
    else
        title(['C' num2str(i) ]);
    end
    xlim([0 150]);
    ylim([-3 3]);
    hold on
    fill([12 30 30 12],[-3 -3 3 3],'g','EdgeColor','none','FaceAlpha',0.3);
    fill([70 150 150 70],[-3 -3 3 3],'y','EdgeColor','none','FaceAlpha',0.3)
end
saveas(gcf,[name '_logpsd'],'fig');
set(gcf,'units','normalized','outerposition',[0 0 1 1])
saveas(gcf,[name '_logpsd'],'fig');

x2= max(max(log10(psd_norm_all)));
x1 = min(min(log10(psd_norm_all)));

log_psd_norm = log10(psd_norm_all);
% figure plot norm PSD
figure;
for i = 1:size(psd_all,2)
    subplot(3,14,i)
    
    ha = gca;
    hold(ha,'on');
    plot(F,log_psd_norm(:,i),'k','LineWidth',2);
    if i==1;
        title(['C' num2str(i) ]); % allows title to have file name
        xlabel('frequency (Hz)');
        ylabel('log PSD');
    else
        %         axis off
        title(['C' num2str(i) ]);
    end
    xlim([0 150]);
    ylim([x1 x2]);
    hold on
    fill([12 30 30 12],[x1 x1 x2 x2],'g','EdgeColor','none','FaceAlpha',0.3);
    fill([70 150 150 70],[x1 x1 x2 x2],'y','EdgeColor','none','FaceAlpha',0.3)
end
saveas(gcf,[name '_normpsd'],'fig');
set(gcf,'units','normalized','outerposition',[0 0 1 1])
saveas(gcf,[name '_normpsd'],'fig');


% figure
% subplot(2,4,1)
% imagesc(log10(psd_delta))
% % caxis([0 1])
% title('delta')
% subplot(2,4,2)
% imagesc(log10(psd_theta))
% % caxis([0 1])
% title('theta')
% subplot(2,4,3)
% imagesc(log10(psd_alpha))
% % caxis([0 1])
% title('alpha')
% subplot(2,4,4)
% imagesc(log10(psd_beta))
% % caxis([0 1])
% title('beta')
% subplot(2,4,5)
% imagesc(log10(psd_Lbeta))
% % caxis([0 1])
% title('Lbeta')
% subplot(2,4,6)
% imagesc(log10(psd_Hbeta))
% % caxis([0 1])
% title('Hbeta')
% subplot(2,4,7)
% imagesc(log10(psd_gamma))
% % caxis([0 1])
% title('gamma')
%
% saveas(gcf,[name(1:end-5) '_psd'],'fig');
