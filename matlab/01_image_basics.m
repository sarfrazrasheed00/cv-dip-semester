% Lesson 01 MATLAB equivalent: image matrix and pixel intensity.
I = uint8([0 128 255; 50 100 200; 25 150 230]);
disp(I);
disp(size(I));
disp(I(2,2)); % MATLAB uses 1-based indexing
imshow(I, 'InitialMagnification', 'fit');
title('3x3 grayscale image');
