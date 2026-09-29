function [target_list_right,target_list_left,RightMask,LeftMask,yn_flip] = define_mask(dlcvideo)

% start with first frame of video
RGB = uint8(dlcvideo(:,:,:,1));

% start by drawing foreground/background boxes to identify the screen -
% changes in intensity in that area will determine when the stimuli are
% on/off. then I use frames when the stimuli are on for the real
% image segmentation
fig1 = figure();
imshow(RGB)
title('Draw a rectangle within the screen')
L = superpixels(RGB,500);
f1 = drawrectangle('Color','g');
foreground = createMask(f1,RGB);
fig2 = figure();
imshow(RGB)
title('Draw a rectangle indicating the background')
b1 = drawrectangle(gca,'Color','r');
background = createMask(b1,RGB);

% this shows what is identified as the screen - precision doesn't matter
% too much, change in intensity is what matters
BW = lazysnapping(RGB,L,foreground,background);
delete(fig1);
delete(fig2);

% figure();
% imshow(labeloverlay(RGB,BW,'Colormap',[0 1 0]))

% now we look at changes in intensity over the first 1000 frames (could be
% even less) to pull out the first trial's start and end frames
mint = zeros(size(dlcvideo,4),1);
for curr_tr=1:size(dlcvideo,4)
    maskedRgbImage = bsxfun(@times, dlcvideo(:,:,:,curr_tr), cast(BW, 'like', dlcvideo(:,:,:,curr_tr)));
    mint(curr_tr)=mean(maskedRgbImage(:));
end

% figure();
% plot(mintensity)
% yline(int_th,'k','linewidth',2)

mintensity = mint - mean(mint);
int_th = -(max(abs(mintensity))-0.4);
% int_th = -0.6;

% make cells with the frames for each trial in im_on
side_on = find(mintensity<int_th);
if length(side_on)==length(mintensity)
    side_on = [];
end

im_on = cell(1,10); k = 1;
for i = 1:5
    im_on{k} = [im_on{k} side_on(i)];
    if side_on(i+1)-side_on(i)>1
        k = k+1;
    end
end

imlen = cellfun(@(x) length(x),im_on)>1;
im_on(~imlen) = [];

% this is the real image we will work with now that we are using frames
% with clearly defined images on the screen
ex_tr = 1;
I = uint8(mean(dlcvideo(:,:,:,im_on{ex_tr}),4));
% figure();
% imshow(I)

% binarize the image with imflatfield and imadjust
sigma = 30;
filtthresh = 140;

% now the user can click the middle of the two stimuli so I can extract the
% objects with centroids closest to where we click
targets_good='N';
num_targetpoints_to_define = 2;
while ~strcmp(targets_good,'Y')
    grayI = rgb2gray(I);
    J = imadjust(imflatfield(grayI,sigma,'FilterSize',115))<filtthresh;
    
    % now use this function regionprops which extracts statistics for "objects"
    % in the image - I extracted the area of the object, the centroid, and
    % extrema
    stats = regionprops('table',J,'Area','Centroid','Extrema','ConvexHull');
    
    % only look at objects with an area > 2 (it will extract single pixels so
    % want to make sure noise isn't confusing it
    stats2 = stats(stats.Area>2,:);
    centroids = cat(1,stats.Centroid);
    
    clf('reset')
    fig3 = figure();
    imagesc(uint8(mean(dlcvideo(:,:,:,im_on{ex_tr}),4))); %display mean image across entire movie
    title('Click on image to identify left then right stimulus.')
    [target_x,target_y]=ginput(num_targetpoints_to_define);
    target_colors=jet(length(target_x));
    hold on
    % some of the earlier videos were flipped horizontally when uploaded,
    % but all recent videos should be fine, i.e. input 'Y'
    yn_flip = input('Is the video flipped correctly? Y/N: ','s');
    if strcmp(yn_flip,'N')
        target_xlist = target_x([2 1],:);
        target_ylist = target_y([2 1],:);
    else
        target_xlist = target_x;
        target_ylist = target_y;
    end
    
    % identify which objects to extract based on centroid/click location
    cent = zeros(1,num_targetpoints_to_define);
    pgon = {};
    for i = 1:num_targetpoints_to_define
        dist = sqrt((stats2.Centroid(:,1) - target_xlist(i)).^2 + (stats2.Centroid(:,2) - target_ylist(i)).^2);
        cent(i) = find(dist == min(dist));
        % convert extrema to polygons to plot
        pgon{i} = polyshape(stats2.ConvexHull{cent(i)}(:,1),stats2.ConvexHull{cent(i)}(:,2));
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

% figure()
% imshow(J)
% hold on;
% plot(pgon{1})
% plot(pgon{2})

% figure()
% imshow(J)

% converts polygon covering left stimulus to mask
LeftMask = poly2mask(stats2.ConvexHull{cent(1)}(:,1),stats2.ConvexHull{cent(1)}(:,2),size(J,1),size(J,2));

% extracts stimulus from video using mask
frames_avg = uint8(mean(dlcvideo(:,:,:,im_on{ex_tr}),4));
maskedRgbImage = bsxfun(@times, frames_avg, cast(LeftMask, 'like', frames_avg));

I=rgb2gray(maskedRgbImage);

% This function finds the corners of the image.
% This seems robust enough
% Across more frames if we see it doesn't hold
% we could set more points on the corner 
% and/or improve accuracy by using mouse click to set corners (as done in other matlab script) 
corners = detectHarrisFeatures(LeftMask,'MinQuality',0.05);
target_list_left = corners.selectStrongest(4).Location;
% figure;
% imshow(I); hold on;
% plot(corners.selectStrongest(4));

% converts polygon covering right stimulus to mask
RightMask = poly2mask(stats2.ConvexHull{cent(2)}(:,1),stats2.ConvexHull{cent(2)}(:,2),size(J,1),size(J,2));

% extracts stimulus from video using mask
frames_avg = uint8(mean(dlcvideo(:,:,:,im_on{ex_tr}),4));
maskedRgbImage = bsxfun(@times, frames_avg, cast(RightMask, 'like', frames_avg));

I=rgb2gray(maskedRgbImage);

% figure()
% imshow(I)

% This function finds the corners of the image.
% This seems robust enough
% Across more frames if we see it doesn't hold
% we could set more points on the corner 
% and/or improve accuracy by using mouse click to set corners (as done in other matlab script) 
corners = detectHarrisFeatures(RightMask);
% figure;
% imshow(I); hold on;
% plot(corners.selectStrongest(4));

target_list_right = corners.selectStrongest(4).Location;

end