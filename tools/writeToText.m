function [ output_args ] = writeToText( filename,images,avgP,avgS,avgTime,dataset)
%WRITETOTEXT �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    sz = length(images);
    T = struct2table(images);
    writetable(T,filename);
    [num, text, raw] = xlsread(filename);
    [rowN, columnN]=size(raw);
    sheet=1;
    xlsRange=['A',num2str(rowN+2)];
    title = [{'DataSet'},{'avg_PSNR'},{'avg_SSIM'},{'avg_TIME'}];
    xlswrite(filename,title,sheet,xlsRange);
    data = [{dataset},{avgP},{avgS},{avgTime}];
    xlsRange=['A',num2str(rowN+3)];
    xlswrite(filename,data,sheet,xlsRange);
end

