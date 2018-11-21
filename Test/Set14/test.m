clc;
clear;
close all;

image=im2double(rgb2gray(imread('Test\Set14\foreman.bmp')));

imshow(image,'Border','tight');
rectangle('Position',[185 178 4 4],'EdgeColor',[1 0 0]);

imwrite(image(178:181,185:188),'test.bmp');