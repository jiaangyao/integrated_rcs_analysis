%% Function Name: transformMVtoRCS()
%
% Description: Converts data in units of mV to the internal unit 
% representation used on the RC+S device.
%
% Inputs:
%     data_mv : (num_samples, num_channels) array, or transpose
%         Data, either time-domain or FFT amplitude, given in units of mV.
%         The result will be returned in the same shape.
%     amp_gain : int, 
%         Parameter indicating the channel gain represented by the
%         cfg_config_data.dev.HT_sns_ampX_gain250_trim value in the
%         DeviceSettings.json file, or the metaData.ampGains OpenMind
%         output.
%
% Outputs:
%     data_rcs : (num_samples, num_channels) array, or transpose
%         Data, either time-domain or FFT amplitude, given in internal RC+S
%         units. Returned in the same shape as the input data.
%
% Author: Tanner Chas Dixon, tanner.dixon@ucsf.edu. Credit to Juan Anso for
%             earlier version of the code.
% Date last updated: February 10, 2022
%---------------------------------------------------------

function data_rcs = transformMVtoRCS(data_mv, amp_gain)
rcs_constant = 48644.8683623726;    % unique RC+S constant
amp_gain = 250*(amp_gain/255);  % convert actual channel amp gain
data_rcs = data_mv * (amp_gain*rcs_constant) / (1000*1.2);
end