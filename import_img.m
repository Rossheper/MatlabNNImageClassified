function out = import_img(fileToRead1)
%IMPORTFILE(FILETOREAD1)
%  Imports data from the specified file
%  FILETOREAD1:  file to read

%  Auto-generated by MATLAB on 11-Mar-2020 12:36:56

% Import the file
newData1 = importdata(fileToRead1);
out = newData1.cdata;
% % Create new variables in the base workspace from those fields.

% for i = 1:length(vars)
%     assignin('base', vars{i}, newData1.(vars{i}));
% end
