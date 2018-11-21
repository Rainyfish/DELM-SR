function  [avgP,avgS] =SR_demo(lambda,NumNodes,path,DataName,pattern)

scale =2 ;

load(['parameters1\dt_291' num2str(scale)]);

load([path '\InputWeight_lambda_' num2str(lambda) '_nodes_' num2str(NumNodes) '.mat']);
load([path '\BiasofHiddenNeurons_lambda_' num2str(lambda) '_nodes_' num2str(NumNodes),'.mat']);
load([path '\OutputWeight_lambda_' num2str(lambda) '_nodes_' num2str(NumNodes),'.mat']);


test_path =[ '/' DataName '/'];
imgPath = ['Test' test_path];
save_path = ['single_layer_gen_image' test_path]% the path of generating HR images

if(~exist(save_path))
    mkdir(save_path);
end
filename = [save_path,DataName,'_',num2str(scale),'.xls'];
xlswrite(filename,[{'imageName'},{'PSNR_BI'},{'SSIM_BI'},{'PSNR'},{'SSIM'},{'Time'}]);

imgDir  = dir([imgPath '*' pattern]); 
%images =();

avgP =0;
avgS = 0;
avgT =0;
time = 0;
for i = 1:length(imgDir)         
    img = im2double(imread([imgPath imgDir(i).name]));
    images(i,1).name = imgDir(i).name;
    
    image = img;
    
    H_15 = [2 8 3 12 10 1 4 11 14 6 9 15 7 13 5];
    
    image = modcrop(image,scale); % crop
    
    h = fspecial('gaussian', 5, 1.6);  % the Gaussian filter
    image_gauss = imfilter( image, h);
    
    sz1 = size(image);
    
    if(size(sz1,2)==2)
        imageL = imresize(image_gauss,1/scale,'bicubic');
        imageB = imresize(imageL,scale,'bicubic');
    else
        image_ycbcr = rgb2ycbcr(image_gauss);
        
        image_y  = im2double(image_ycbcr(:,:,1));
        image_cb = im2double(image_ycbcr(:,:,2));
        image_cr = im2double(image_ycbcr(:,:,3));
        
        imageL    = imresize(image_y,1/scale,'bicubic');
        imageL_cb = imresize(image_cb,1/scale,'bicubic');
        imageL_cr = imresize(image_cr,1/scale,'bicubic');
        
        imageB = zeros(size(image_ycbcr));
        imageB(:,:,1) = imresize(imageL,scale,'bicubic');
        imageB(:,:,2) = imresize(imageL_cb,scale,'bicubic');
        imageB(:,:,3) = imresize(imageL_cr,scale,'bicubic');
        
        imageH_rec = zeros(size(image_ycbcr));
        imageH_rec(:,:,2) = imageB(:,:,2);
        imageH_rec(:,:,3) = imageB(:,:,3);
    end
    
    
    imageH=zeros(scale*size(imageL));
    
    H_16=hadamard( 16 );
    
    H_16(:,1) =[];
    
    
    sz = size(imageL);
    imagepadding = zeros(sz(1)+2,sz(2)+2);
    imagepadding(2:end-1,2:end-1) = imageL;
    
    offset = floor( scale / 2 );
    startt = tic;
    [imageH]= SR_2_ELM( imagepadding, dt, H_16, InputWeight, BiasofHiddenNeurons, OutputWeight,NumNodes);
    current_time = toc(startt);
    time=time+current_time;
    
    if(size(sz1,2)==2)
        imageH_rec = imageH;
    else
        imageH_rec(:,:,1) = imageH;
        imageB = ycbcr2rgb( imageB );
        imageH_rec = ycbcr2rgb( imageH_rec );
    end
    %imshow(imageH_rec);
    if(mod(scale,2) == 0)
        if(size(sz1,2)==2)
            imageB = imageB( offset + scale + 2 + 3 : end - ( offset + scale + 1  + 3 ),...
                offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ));
            imageH_rec = imageH_rec( offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),...
                offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ));
            image  = image(offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),...
                offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ));
            
        else
            imageB = imageB( offset + scale + 2 + 3 : end - ( offset + scale + 1  + 3 ),...
                offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),:);
            imageH_rec = imageH_rec( offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),...
                offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),:);
            image  = image(offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),...
                offset + scale + 2  + 3 : end - ( offset + scale + 1  + 3 ),:);
            
        end
        
        [p1,s1] = compute_psnr_ssim(image,imageB); % Bicubic
        [p2,s2] = compute_psnr_ssim(image,imageH_rec); % Our
        
        
        
    else
        if(size(sz1,2)==2)
            imageB = imageB( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
                offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ) );
            imageH_rec = imageH_rec( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
                offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ) );
            image  = image( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
                offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ) );
            
            
        else
            imageB = imageB( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
                offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),:);
            imageH_rec = imageH_rec( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
                offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),:);
            image  = image( offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),...
                offset + scale + 2  + 5 : end - ( offset + scale + 2  + 5 ),:);
            
            
        end
        
        [p1,s1] = compute_psnr_ssim(image,imageB); % Bicubic
        [p2,s2] = compute_psnr_ssim(image,imageH_rec); % Our
        
        
    end
    images(i,1).bicPSNR=p1;
    images(i,1).ourPSNR =p2;
    images(i,1).bicSSIM = s1;
    images(i,1).ourSSIM = s2;
    images(i,1).ourSSIM = s2;
    images(i,1).Time = current_time;
    %
    
    
   imwrite(imageH_rec,[save_path,imgDir(i).name,'_our_ELM1_x',num2str(scale),'.',pattern]);
    display(['image name **  ' imgDir(i).name])
    display(['Bicubic PSNR ' num2str(p1)]);
    display(['ELM     PSNR ' num2str(p2)]);
    display(['Bicubic SSIM ' num2str(s1)]);
    display(['ELM     SSIM ' num2str(s2)]);
    avgP = avgP+p2;
    avgS = avgS+s2;
    
    
    %end
end
addpath('tools')
avgP = avgP/length(imgDir);
avgS = avgS/length(imgDir);
avgTime = time/length(imgDir);
writeToText(filename,images,avgP,avgS,avgTime,DataName);
display(['avg PSNR ' num2str(avgP)]);
display(['avg SSIM ' num2str(avgS)]);
display(['avg time ' num2str(avgTime)]);
end
