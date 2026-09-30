function [shrewMask_side,shrewMask_cent] = mask_shrew(dlcFrame_side,dlcFrame_cent)

I = uint8(dlcFrame_cent{1}(:,:,:,1));

% binarize the image with imflatfield and imadjust
sigma = 30;
filtthresh = 150;

% now the user can click the middle of the two stimuli so I can extract the
% objects with centroids closest to where we click
targets_good='N';
num_targetpoints_to_define = 1;

% now the user can click the middle of the two stimuli so I can extract the
% objects with centroids closest to where we click
clf('reset')
fig3 = figure();
imagesc(I); %display image from frame
title('Click on image to identify shrew.')
[target_x,target_y]=ginput(num_targetpoints_to_define);
close(fig3);

while ~strcmp(targets_good,'Y')
    % clf('reset')
    % fig3 = figure();
    % imagesc(I); %display mean image across entire movie
    % hold on;

    grayI = rgb2gray(I);
    J = imadjust(imflatfield(grayI,sigma,'FilterSize',115))<filtthresh;
    J = J(2:end-1,2:end-1);
    J = padarray(J,[1,1], 0, 'both');

    % figure;
    % imshow(J)
    
    % now use this function regionprops which extracts statistics for "objects"
    % in the image - I extracted the area of the object, the centroid, and
    % extrema
    stats = regionprops('table',J,'Area','Centroid','Extrema','ConvexHull');
    
    % only look at objects with an area > 2 (it will extract single pixels so
    % want to make sure noise isn't confusing it
    stats_shrew = stats(stats.Area>10,:);

    % figure;
    % imshow(J)
    % hold on;
    % for k = 1:height(stats_shrew)
    %     hull = stats_shrew(k,:).ConvexHull{1};
    %     plot(hull(:,1), hull(:,2), 'r-', 'LineWidth', 2);
    %     hold on;
    % end
    
    % identify which objects to extract based on centroid/click location
    dist = sqrt((stats_shrew.Centroid(:,1) - target_x).^2 + (stats_shrew.Centroid(:,2) - target_y).^2);
    % will plot masks on top of video frame - make sure they cover the full
    % stimulus area, if not, input 'N' and try a new threshold
    % Build a mask of only the binary pixels inside each selected convex hull
    [imgH, imgW] = size(J);
    
    hull = stats_shrew.ConvexHull{dist == min(dist)};   % Nx2 [x y] vertices

    % Rasterize the hull polygon into a logical mask
    hullMask = poly2mask(hull(:,1), hull(:,2), imgH, imgW);

    % Keep only the part of the binary image inside the hull
    masked_regions = J & hullMask;

    L = bwlabel(masked_regions, 8);     % 8-connectivity; use 4 for stricter connectivity

    lbl = L(round(target_y), round(target_x));   % note: row = y, col = x
    
    if lbl > 0
        finalMask = (L == lbl);
    else
        warning('Target point is not on the mask; falling back to largest region.');
        finalMask = bwareafilt(masked_regions, 1);
    end
    finalMask = imfill(finalMask, 'holes');
    
    clf('reset')
    fig3 = figure();
    imshow(labeloverlay(I, finalMask, 'Colormap', [1 0 0], 'Transparency', 0.6));
    title('Binary mask inside convex hull');
    
    targets_good=input('Do the targets look good? Y/N: ','s');
    if ~strcmp(targets_good,'Y')
        fprintf('Redoing target definition \n')
        filtthresh = input(['New filter threshold (previously ',num2str(filtthresh),'): ']);
        close(fig3);
    else
        fprintf('Targets are good. \n')
    end
end

shrewMask_side = nan(size(dlcFrame_side{1},1),size(dlcFrame_side{1},2),length(dlcFrame_side));
for i = 1:length(dlcFrame_side)

    I = dlcFrame_side{i}(:,:,:,round(end/2));
    grayI = rgb2gray(I);
    J = imadjust(imflatfield(grayI,sigma,'FilterSize',115))<filtthresh;
    J = J(2:end-1,2:end-1);
    J = padarray(J,[1,1], 0, 'both');

    % now use this function regionprops which extracts statistics for "objects"
    % in the image - I extracted the area of the object, the centroid, and
    % extrema
    stats = regionprops('table',J,'Area','Centroid','Extrema','ConvexHull');

    % only look at objects with an area > 2 (it will extract single pixels so
    % want to make sure noise isn't confusing it
    stats_shrew = stats(stats.Area>50,:);

    % identify which objects to extract based on centroid/click location
    dist = sqrt((stats_shrew.Centroid(:,1) - target_x).^2 + (stats_shrew.Centroid(:,2) - target_y).^2);

    % will plot masks on top of video frame - make sure they cover the full
    % stimulus area, if not, input 'N' and try a new threshold
    % Build a mask of only the binary pixels inside each selected convex hull
    [imgH, imgW] = size(J);
    hull = stats_shrew.ConvexHull{dist == min(dist)};   % Nx2 [x y] vertices

    % Rasterize the hull polygon into a logical mask
    hullMask = poly2mask(hull(:,1), hull(:,2), imgH, imgW);

    % Keep only the part of the binary image inside the hull
    masked_regions = J & hullMask;

    L = bwlabel(masked_regions, 8);     % 8-connectivity; use 4 for stricter connectivity
    lbl = L(round(target_y), round(target_x));   % note: row = y, col = x
    
    if lbl > 0
        finalMask = (L == lbl);
    else
        warning('Target point is not on the mask; falling back to largest region.');
        finalMask = bwareafilt(masked_regions, 1);
    end
    finalMask = imfill(finalMask, 'holes');
    shrewMask_side(:,:,i) = finalMask;

end

shrewMask_cent = nan(size(dlcFrame_cent{1},1),size(dlcFrame_cent{1},2),length(dlcFrame_cent));
for i = 1:length(dlcFrame_cent)

    I = dlcFrame_cent{i}(:,:,:,round(end/2));
    grayI = rgb2gray(I);
    J = imadjust(imflatfield(grayI,sigma,'FilterSize',115))<filtthresh;
    J = J(2:end-1,2:end-1);
    J = padarray(J,[1,1], 0, 'both');

    % now use this function regionprops which extracts statistics for "objects"
    % in the image - I extracted the area of the object, the centroid, and
    % extrema
    stats = regionprops('table',J,'Area','Centroid','Extrema','ConvexHull');

    % only look at objects with an area > 2 (it will extract single pixels so
    % want to make sure noise isn't confusing it
    stats_shrew = stats(stats.Area>50,:);

    % identify which objects to extract based on centroid/click location
    dist = sqrt((stats_shrew.Centroid(:,1) - target_x).^2 + (stats_shrew.Centroid(:,2) - target_y).^2);

    % will plot masks on top of video frame - make sure they cover the full
    % stimulus area, if not, input 'N' and try a new threshold
    % Build a mask of only the binary pixels inside each selected convex hull
    [imgH, imgW] = size(J);
    hull = stats_shrew.ConvexHull{dist == min(dist)};   % Nx2 [x y] vertices

    % Rasterize the hull polygon into a logical mask
    hullMask = poly2mask(hull(:,1), hull(:,2), imgH, imgW);

    % Keep only the part of the binary image inside the hull
    masked_regions = J & hullMask;

    L = bwlabel(masked_regions, 8);     % 8-connectivity; use 4 for stricter connectivity
    lbl = L(round(target_y), round(target_x));   % note: row = y, col = x
    
    if lbl > 0
        finalMask = (L == lbl);
    else
        warning('Target point is not on the mask; falling back to largest region.');
        finalMask = bwareafilt(masked_regions, 1);
    end
    finalMask = imfill(finalMask, 'holes');
    shrewMask_cent(:,:,i) = finalMask;

end

end