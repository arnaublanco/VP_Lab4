close all;
clearvars;
slic_dir = '../SLIC_mex/';
addpath(slic_dir);

directoryIm = '../datasets/images/';
dataSet = 'ambush_2';

I1 = double(imread(fullfile(directoryIm, dataSet, 'frame_0001.png'))) / 256;
I2 = double(imread(fullfile(directoryIm, dataSet, 'frame_0002.png'))) / 256;

directoryOF = '../datasets/flow/';
% Optical flow vector
OFGT = readFlowFile (fullfile(directoryOF, dataSet, 'frame_0001.flo'));

directoryOcc = '../datasets/occlusions/';
% Ground truth occlusion
OccGT = double(imread(fullfile(directoryOcc, dataSet, 'frame_0001.png'))) > 128;

nFig=0;
nFig=nFig+1;
figure(nFig)
imshow(I1);

nFig=nFig+1;
figure(nFig)
imshow(I2);

%% Step 1 and 2: Build xi_1 and eta_12

close all;
% Build warping function
I1_from_I2 = warping(I1, I2, OFGT(:,:,1), OFGT(:,:,2));

nFig = nFig + 1;
figure(nFig)
imshow(I1_from_I2);

sigma_spatial = 1;
sigma_grayLevel = 0.1;

eps_g =  1e-3;
sigma = [sigma_spatial, sigma_grayLevel];

winSize = 3; % Window size
[ni, nj, nC] = size(I1);
eta_12 = zeros(ni,nj,nC);

% Compute the images xi_1 and eta_12 using the cross bilateral filter.
xi_1 = bfilter2(I1, winSize, sigma);
for n = 1:nC
    eta_12(:,:,n) = cross_bilateral_filter(I1(:,:,n), I1_from_I2(:,:,n), winSize, sigma);
end

nFig = nFig+1;
figure(nFig)
imshow(xi_1);

nFig = nFig+1;
figure(nFig)
imshow(eta_12);

%% Step 3: Oversegmentation

[ni, nj, nC] = size(xi_1);

weKeep = 0.05; % in percentage
nLabels = round(ni*nj*weKeep/100);

%numSuperpixels is the same as number of superpixels.
[lblP, numSuperpixels] = slicmex(uint8(xi_1*256),nLabels,20);

lblP = double(lblP)+1; %We want labels from 1 to numSuperpixels

% To visualize purpose
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(lblP), hy, 'replicate');
Ix = imfilter(double(lblP), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
idx = gradmag >0;

xi_1_toShow_r = xi_1(:,:,1);
xi_1_toShow_g = xi_1(:,:,2);
xi_1_toShow_b = xi_1(:,:,3);

xi_1_toShow_r(idx) = 1;
xi_1_toShow_g(idx) = 0;
xi_1_toShow_b(idx) = 0;

xi_1_toShow(:,:,1) = xi_1_toShow_r;
xi_1_toShow(:,:,2) = xi_1_toShow_g;
xi_1_toShow(:,:,3) = xi_1_toShow_b;
% end to visualize purpose

nFig = nFig+1;
figure(nFig)
imshow(xi_1_toShow);
%hold on

nFig = nFig+1;
figure(nFig)
imagesc(lblP); colorbar;
axis off

%% Step 4: Gaussian Mixture Estimation

nDist = 2; % Number of gaussians
GM = cell(int16(numSuperpixels),1);

%Fit a gaussian for each superpixel
for n=1:numSuperpixels

    %Select all pixels belonging the same superpixel
     
    
    %data =  ;%pixels of xi_1 belonging to superpixel n
    
    try
        %Model the gaussian mixture model of the data
        GM{n} = fitgmdist(data, nDist);
    catch exception
        disp('There was an error fitting the Gaussian mixture model')
        error = exception.message;
        disp(error)
        GM{n} = fitgmdist(data,nDist,'Regularize',0.1);
    end
    %plot3(data(:,1), data(:,2), data(:,3), '*')
    %pause
end

%% Step 5: soft-occlusion map

softMap=zeros(ni,nj);
for n=1:numSuperpixels
    
   %Select all pixels belonging the same superpixel
 
    %data = ;%pixels of eta_12 belonging to superpixel n
    
    %Probability of belonging the gmm of the superpixel
    postProb = pdf(GM{n}, data);
    
    p=-log(postProb);
    
    softMap(mask)=p;
end

nFig=nFig+1;
figure(nFig)
imagesc(softMap); colorbar

%% Step 6: Hard occlusion map (threshold)

thr =0; %decision threshold
hardMap = softMap>thr; 

nFig=nFig+1;
figure(nFig)
imagesc(hardMap) %Estimated occlusion

%% Step 7: Comparison against ground truth

nFig=nFig+1;
figure(nFig)
imagesc(OccGT) %Ground truth occlusion
