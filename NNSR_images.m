close all
clear
clc
PSNR = [];
peaksnr_0=[];
peaksnr_1=[];
t_0=[];
t_1=[];
RMSE = [];
RMSE_0 = [];
RMSE_1 = [];
PSNR_1 = [];
SSIM_1 = [];
maxiter = 100;
tol     = 1e-5;

image = imread('1.jpg');
[width,height,z]=size(image);
if(z>1)
    image=rgb2gray(image);
end
% unit8 to double
image = mat2gray(image);
[m,n] = size(image);
M = (image);

for kk = 1:20

%% random mask
per = 0.8;
array_Omega = binornd( 1, per, [ m, n ] );

AM = 4;
ps = 0.2;
EL0=round(m*n*ps);
E=zeros(m,n);
Ind = randperm(m*n);
E(Ind(1:EL0))=AM*(rand(1,EL0)-0.5);
M_noise = M + E; 

M_noise = M_noise.* array_Omega;

%% NNSR

addpath('lansvd');
tic
[X_1,E_Welsch,RE_M3, RE_S3,RE_X3,RE_E3,RE_inf] = NNSR_RMC(M_noise,array_Omega,M,array_Omega,sqrt(2),sqrt(2),1.5);
toc
t_1 = [t_1 toc];
RMSE_1 = [RMSE_1 norm((M-X_1),'fro')/norm(M,'fro')];
PSNR_1 = [PSNR_1 psnr(X_1,M)];
SSIM_1 = [SSIM_1 ssim(X_1,M)];


end


