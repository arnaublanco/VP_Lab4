function [I2_warped] = warping(I1, I2, u, v)
% Inputs:
%   - I1, I2, input images
%   - u, v: vectors optical flow

% Output:
%   - I2_warped: image warped I1 from I2

[M, N, C] = size(I1);

% Take the pixel positions of I1 and apply u,v, taking into account the borders
idxx = zeros(M,N);
idyy = zeros(M,N);

for i = 1:M
    for j = 1:N
        ju = j-u(i,j);
        iv = i-v(i,j);
        if ju <= 0
            ju = 1;
        elseif ju > N
            ju = N;
        end
        if iv <= 0
            iv = 1;
        elseif iv > M
            iv = M;
        end
        idxx(i,j) = ceil(ju);
        idyy(i,j) = ceil(iv);
    end
end

% Warp the image in each channel
for i = 1:C
    I2_warped(:,:,i) = interp2(I2(:,:,i),idxx,idyy,'cubic');
end