function hann_win = hannWindow(L,percentage)
% calcualtes hann window given window size L and percentage (25%,50%,100%)
% input
% (1) L: window size (integer)
% (2) Percentage: '25% Hann', '50% Hann' or '100% Hann' (string) (default, '100% Hann')

if nargin ==0
    error('you need to provide at least the number of window samples L')
elseif nargin ==1
    percentage = '100% Hann';
elseif nargin > 2
    error('too many input arguments')
end
     
switch percentage
    case '100% Hann'
        hann_win = 0.5*(1-cos(2*pi*(0:L-1)/(L-1))); % create hann taper function, equivalent to the Hann 100%         
    case '50% Hann'
        temp_win = 0.5*(1-cos(4*pi*(0:L-1)/(L-1))); % create hann taper function, equivalent to the Hann 50% 
        hann_win = sethannwindow(temp_win);                
    case '25% Hann'
        temp_win = 0.5*(1-cos(8*pi*(0:L-1)/(L-1))); % create hann taper function, equivalent to the Hann 250% 
        hann_win = sethannwindow(temp_win);  
    case '0% Hann'
        hann_win = ones([1,L]);
end

end

function hann_out = sethannwindow(temp_win)
    % sets hann window flat top in case is not at 100% hann window 
    [~,indeces] = findpeaks(temp_win);
    temp_win(indeces(1)+1:indeces(end)-1) = 1;
    hann_out = temp_win;        
end
        