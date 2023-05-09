% use for finding trial onset for old Tap That Emotion (1 beep per trial)
% finds beep, which signals end of trial
% take these indices and subtract RT for trial onset

function [trial_onset]=find_trial_onset(data,cue_threshold)

[peakheight peakinds] = findpeaks(data,'MinPeakHeight',cue_threshold);

% find all beeps
% identifies indices for start of each cue
event=peakinds(1);
for i=1:(length(peakinds)-1)
    if (peakinds(i+1)-peakinds(i)>2000)
        event=[event peakinds(i+1)];
    end
end

trial_onset=event;

figure; plot(data); hold on; plot(trial_onset,cue_threshold,'*r');

