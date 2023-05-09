function ciplot(lower,upper,x,colour,facealpha,edgealpha)


if any(isnan(lower))||any(isnan(upper))
    warning('There are nans in the data... removing for ciplot');
    deleteVec = isnan(lower)|isnan(upper);
    lower(deleteVec) = [];
    x(deleteVec) = [];
    upper(deleteVec) = [];
end

% ciplot(lower,upper)
% ciplot(lower,upper,x)
% ciplot(lower,upper,x,colour)
%
% Plots a shaded region on a graph between specified lower and upper confidence intervals (L and U).
% l and u must be vectors of the same length.
% Uses the 'fill' function, not 'area'. Therefore multiple shaded plots
% can be overlayed without a problem. Make them transparent for total visibility.
% x data can be specified, otherwise plots against index values.
% colour can be specified (eg 'k'). Defaults to blue.

% Raymond Reynolds 24/11/06
% Modified by me, to add transparency.
if length(lower)~=length(upper)
    error('lower and upper vectors must be same length')
end
usealpha = 0;
if nargin<6
    edgealpha = 0;
end

if nargin<5
    facealpha=1;
end

if nargin<4
    colour='b';
end

if nargin<3
    x=1:length(lower);
end

if nargin >= 5
    usealpha = 1;
end
% convert to row vectors so fliplr can work
if find(size(x)==(max(size(x))))<2
    x=x'; end
if find(size(lower)==(max(size(lower))))<2
    lower=lower'; end
if find(size(upper)==(max(size(upper))))<2
    upper=upper'; end
if (facealpha==1)
    usealpha = 0;
end
if usealpha == 1
    fill([x fliplr(x)],[upper fliplr(lower)],colour,'EdgeColor','none','facealpha',facealpha,'edgealpha',edgealpha);
else
    fill([x fliplr(x)],[upper fliplr(lower)],colour,'EdgeColor','none');
end
