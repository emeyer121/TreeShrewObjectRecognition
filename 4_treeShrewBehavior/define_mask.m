function [stats_side, stats_cent, mid_side, mid_cent] = define_mask(vidPath, image)

fig1 = figure();
imshow(image)
title('Draw a rectangle within the screen')
L = superpixels(image,500);
f1 = drawrectangle('Color','g');
foreground = createMask(f1,image);
fig2 = figure();
imshow(image)
title('Draw a rectangle indicating the background')
b1 = drawrectangle(gca,'Color','r');
background = createMask(b1,image);

% this shows what is identified as the screen - precision doesn't matter
% too much, change in intensity is what matters
BW = lazysnapping(image,L,foreground,background);
delete(fig1);
delete(fig2);

dlcvideo_meta = VideoReader(vidPath);

% now we look at changes in intensity over the first 1000 frames (could be
% even less) to pull out the first trial's start and end frames
frameRange = 1:100;
nframes = length(frameRange);
dlcvideo = nan(size(image,1),size(image,2),size(image,3),nframes);
ii=1;
for curr_tr=1:max(frameRange)
    fr = readFrame(dlcvideo_meta);
    if ismember(curr_tr,frameRange)
        dlcvideo(:,:,:,ii) = fr;
        ii=ii+1;
    end
end
clear dlcvideo_meta

mint = zeros(nframes,1);
for curr_tr=1:nframes
    maskedRgbImage = bsxfun(@times, dlcvideo(:,:,:,curr_tr), cast(BW, 'like', dlcvideo(:,:,:,curr_tr)));
    mint(curr_tr)=mean(maskedRgbImage(:));
end

mintensity = mint - mean(mint);
int_th = -(max(abs(mintensity))-0.5);
% int_th = -0.6;

% make cells with the frames for each trial in im_on
side_on = find(mintensity<int_th);
if length(side_on)==length(mintensity)
    side_on = [];
end

cent_on = find((mintensity>int_th-0.3) & (mintensity<int_th+0.5));

% this is the real image we will work with now that we are using frames
% with clearly defined images on the screen
ex_tr = 1;
I = uint8(mean(dlcvideo(:,:,:,side_on(ex_tr)),4));

% binarize the image with imflatfield and imadjust
sigma = 30;
filtthresh = 140;

% now the user can click the middle of the two stimuli so I can extract the
% objects with centroids closest to where we click
targets_good='N';
num_targetpoints_to_define = 2;

% now the user can click the middle of the two stimuli so I can extract the
% objects with centroids closest to where we click
clf('reset')
fig3 = figure();
imagesc(uint8(dlcvideo(:,:,:,side_on(ex_tr)))); %display mean image across entire movie
title('Click on image to identify left then right stimulus.')
[target_x,target_y]=ginput(num_targetpoints_to_define);
hold on
% some of the earlier videos were flipped horizontally when uploaded,
% but all recent videos should be fine, i.e. input 'Y'
target_xlist = target_x;
target_ylist = target_y;
close(fig3);

while ~strcmp(targets_good,'Y')
    clf('reset')
    fig3 = figure();
    imagesc(uint8(dlcvideo(:,:,:,side_on(ex_tr)))); %display mean image across entire movie
    hold on;

    grayI = rgb2gray(I);
    J = imadjust(imflatfield(grayI,sigma,'FilterSize',115))<filtthresh;
    
    % now use this function regionprops which extracts statistics for "objects"
    % in the image - I extracted the area of the object, the centroid, and
    % extrema
    stats = regionprops('table',J,'Area','Centroid','Extrema','ConvexHull');
    
    % only look at objects with an area > 2 (it will extract single pixels so
    % want to make sure noise isn't confusing it
    stats_side = stats(stats.Area>2,:);
    
    % identify which objects to extract based on centroid/click location
    mid_side = zeros(1,num_targetpoints_to_define);
    pgon = {};
    for i = 1:num_targetpoints_to_define
        dist = sqrt((stats_side.Centroid(:,1) - target_xlist(i)).^2 + (stats_side.Centroid(:,2) - target_ylist(i)).^2);
        mid_side(i) = find(dist == min(dist));
        % convert extrema to polygons to plot
        pgon{i} = polyshape(stats_side.ConvexHull{mid_side(i)}(:,1),stats_side.ConvexHull{mid_side(i)}(:,2));
    end
    % will plot masks on top of video frame - make sure they cover the full
    % stimulus area, if not, input 'N' and try a new threshold
    imshow(J)
    hold on;
    plot(pgon{1})
    plot(pgon{2})
    plot(target_xlist(1),target_ylist(1),'k*')
    plot(target_xlist(2),target_ylist(2),'k*')
    hold on
    targets_good=input('Do the targets look good? Y/N: ','s');
    if ~strcmp(targets_good,'Y')
        fprintf('Redoing target definition \n')
        filtthresh = input(['New filter threshold (previously ',num2str(filtthresh),'): ']);
        close(fig3);
    else
        fprintf('Targets are good. \n')
    end
end

% this is the real image we will work with now that we are using frames
% with clearly defined images on the screen
ex_tr = 1;
I = uint8(mean(dlcvideo(:,:,:,cent_on(ex_tr)),4));

% binarize the image with imflatfield and imadjust
sigma = 30;
filtthresh = 140;

% now the user can click the middle of the two stimuli so I can extract the
% objects with centroids closest to where we click
targets_good='N';
num_targetpoints_to_define = 1;

% now the user can click the middle of the two stimuli so I can extract the
% objects with centroids closest to where we click
clf('reset')
fig3 = figure();
imagesc(uint8(dlcvideo(:,:,:,cent_on(ex_tr)))); %display mean image across entire movie
title('Click on image to identify center stimulus.')
[target_x,target_y]=ginput(num_targetpoints_to_define);
close(fig3);

while ~strcmp(targets_good,'Y')
    clf('reset')
    fig3 = figure();
    imagesc(uint8(dlcvideo(:,:,:,cent_on(ex_tr)))); %display mean image across entire movie
    hold on;

    grayI = rgb2gray(I);
    J = imadjust(imflatfield(grayI,sigma,'FilterSize',115))<filtthresh;
    
    % now use this function regionprops which extracts statistics for "objects"
    % in the image - I extracted the area of the object, the centroid, and
    % extrema
    stats = regionprops('table',J,'Area','Centroid','Extrema','ConvexHull');
    
    % only look at objects with an area > 2 (it will extract single pixels so
    % want to make sure noise isn't confusing it
    stats_cent = stats(stats.Area>20,:);
    
    % identify which objects to extract based on centroid/click location
    mid_cent = zeros(1,num_targetpoints_to_define);
    pgon = {};
    for i = 1:num_targetpoints_to_define
        dist = sqrt((stats_cent.Centroid(:,1) - target_x(i)).^2 + (stats_cent.Centroid(:,2) - target_y(i)).^2);
        mid_cent(i) = find(dist == min(dist));
        % convert extrema to polygons to plot
        pgon{i} = polyshape(stats_cent.ConvexHull{mid_cent(i)}(:,1),stats_cent.ConvexHull{mid_cent(i)}(:,2));
    end
    % will plot masks on top of video frame - make sure they cover the full
    % stimulus area, if not, input 'N' and try a new threshold
    imshow(J)
    hold on;
    plot(pgon{1})
    plot(target_x(1),target_y(1),'k*')
    hold on
    targets_good=input('Do the targets look good? Y/N: ','s');
    if ~strcmp(targets_good,'Y')
        fprintf('Redoing target definition \n')
        filtthresh = input(['New filter threshold (previously ',num2str(filtthresh),'): ']);
        close(fig3);
    else
        fprintf('Targets are good. \n')
    end
end

end