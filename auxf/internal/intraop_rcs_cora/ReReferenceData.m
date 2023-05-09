

function [ecog, lfp, bad,car] = ReReferenceData(ecog, lfp, ref_method, name)
bad=[];
if ref_method <2 % use common reference
    length_CAR = length(ecog.contact);
    data = nan*ones(length_CAR,length(ecog.contact(1).signal));
    % compute common ref and exclude bad channels
    for i = 1: length_CAR
        ecog.contact(i).signal=ecog.contact(i).signal;
        if ~isempty(ecog.contact(i).signal)
            data(i,:) = ecog.contact(i).signal;
        end
        %         if ~isempty(strfind(name,'theta')) || ~isempty(strfind(name,'stim')) || ~isempty(strfind(name,'contin')) || ~isempty(strfind(name,'DBS')) || ~isempty(strfind(name,'Stim')) || ~isempty(strfind(name,'closedLoop'))%s || ~isempty(strfind(name,'OFC'))
        %             bad = [bad];
        %         else
        %
        %             if find(abs(ecog.contact(i).signal)>=800)
        %                 bad = [bad i];
        %             end
        %
        % %             [psd,F] = pwelch(ecog.contact(i).signal,2^(nextpow2(ecog.Fs(i))),2^(nextpow2(ecog.Fs(i)/2)),2^(nextpow2(ecog.Fs(i))),ecog.Fs(i));
        % %             idx_B = find(F>=5 & F<=25);
        % %             idx_G = find(F>65 & F<=100 | F>135 & F<=175);
        % %             if find(psd(idx_G)> mean(psd(idx_B))/2.5)
        % %                 bad=[bad i];
        % %             end
        %         end
        %         bad=unique(bad);
        %     end
        %     good = setdiff(1:length(ecog.contact), bad);
        
        good = [1:length(ecog.contact)];
        
        if ref_method == 0 %  use common  mean as reference
            car=nanmean(data(good,:));
            
        elseif ref_method ==1 % use common median reference
            car=nanmedian(data(good,:));
        end
        for i = 1: length_CAR
            if ~isempty(ecog.contact(i).signal)
                ecog.contact(i).signal_ref = ecog.contact(i).signal-car;
            end
        end
    end
elseif ref_method == 2 % use bipolor reference
    if length(ecog.contact)==28
        for i = 1: length(ecog.contact)/2
            if ~isempty(ecog.contact(i).signal)
                ecog.contact(i).signal_ref = ecog.contact(i).signal-ecog.contact(i+length(ecog.contact)/2).signal;
                ecog.contact(i+length(ecog.contact)/2).signal_ref = ecog.contact(i+length(ecog.contact)/2).signal-ecog.contact(i).signal;
            end
        end
    else
        for i = 1: length(ecog.contact)
            if ~isempty(ecog.contact(i).signal)
                if i <= length(ecog.contact)-1
                    ecog.contact(i).signal_ref = ecog.contact(i).signal-ecog.contact(i+1).signal;
                    %                 else
                    %                     ecog.contact(i).signal_ref = ecog.contact(i).signal;
                end
                
            end
        end
    end
else % no rereferncing
    for i = 1: length(ecog.contact)/2
        if ~isempty(ecog.contact(i).signal)
            ecog.contact(i).signal_ref = ecog.contact(i).signal;
            ecog.contact(i+length(ecog.contact)/2).signal_ref = ecog.contact(i+length(ecog.contact)/2).signal;
        end
    end
end

if ~isempty(lfp)
    for i = 1:length(lfp.contact)-1
        lfp.contact(i).signal_ref=lfp.contact(i).signal-lfp.contact(i+1).signal;
    end
    i = length(lfp.contact);
    lfp.contact(i).signal_ref=lfp.contact(i).signal-lfp.contact(i-1).signal;
end

% % take 1 min
% if ~isempty(lfp)
%
%     for i = 1:length(lfp.contact)
%         if length(lfp.contact(i).signal_ref)>=60000 & length(lfp.contact(i).signal_ref)<=70000
%             lfp.contact(i).signal_ref=lfp.contact(i).signal_ref(1:60000);
%         end
%     end
% end
% for i = 1:length(ecog.contact)
%     if length(ecog.contact(i).signal_ref)>=60000 & length(ecog.contact(i).signal_ref)<=70000
%         ecog.contact(i).signal_ref=ecog.contact(i).signal_ref(1:60000);
%     end
% end