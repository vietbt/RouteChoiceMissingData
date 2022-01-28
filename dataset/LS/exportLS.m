% Exporting data

for i=1:length(LSatt)
    i
    filename = sprintf('LS%d.txt',i);
    LSatt(i).value(find(LSatt(i).value<0.0001)) = 0;
    [i,j,val]=find(LSatt(i).value);
    dlmwrite(filename,[i j val],'delimiter', ' ','newline','pc');
end