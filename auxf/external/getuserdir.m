function userDir = getuserdir
%GETUSERDIR   return the user home directory.
%   USERDIR = GETUSERDIR returns the user home directory using the registry
%   on windows systems and using Java on non windows systems as a string
%
%   Example:
%      getuserdir() returns on windows
%           C:\Documents and Settings\MyName\Eigene Dateien

if ispc
    userDir = winqueryreg('HKEY_CURRENT_USER',...
        ['Software\Microsoft\Windows\CurrentVersion\' ...
         'Explorer\Shell Folders'],'Personal');
     
     %James mod:
     [pathstr_temp,~,~] = fileparts(userDir);
     userDir = pathstr_temp;
     if strcmp(getenv('COMPUTERNAME'),'JAMES-LAPTOP')
         %Because I have it set up stupidly:
         userDir = 'C:\Users\mcintosh';
     elseif strcmp(getenv('COMPUTERNAME'),'DESKTOP-0IIRBHG')
         %Because I have it set up stupidly:
         userDir = 'D:';
     end
else
    userDir = char(java.lang.System.getProperty('user.home'));
end

%add habanero
if strcmpi(userDir,'/rigel/home/jrm2263')
    userDir = '/rigel/dsi/users/jrm2263';
end
end