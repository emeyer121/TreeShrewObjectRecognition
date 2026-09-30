function [dlcFrame_side,dlcFrame_cent,camel_side,imageID] = extract_objects(...
    vidPath,side_on,cent_on,targ_list_right,targ_list_left,targ_list_cent,RightMask,LeftMask)

ir = 1;ic = 1;
num_r = 1;num_c = 1;
dlcvideo_meta = VideoReader(vidPath);
max_frame = dlcvideo_meta.NumFrames;

dlcFrame_side = cell(1,length(side_on));
dlcFrame_cent = cell(1,length(cent_on));
for curr_fr=1:max_frame
    fr = readFrame(dlcvideo_meta);
    if ismember(curr_fr, side_on{ir})
        dlcFrame_side{ir}(:,:,:,num_r) = fr;
        num_r = num_r+1;
    elseif curr_fr > max(side_on{ir})
        ir = ir+1;
        num_r = 1;
        if curr_fr >= max(side_on{end})
            ir = ir-1;
            continue
        end
    end
    if ismember(curr_fr, cent_on{ic})
        dlcFrame_cent{ic}(:,:,:,num_c) = fr;
        num_c = num_c+1;
    elseif curr_fr > max(cent_on{ic})
        ic = ic+1;
        num_c = 1;
        if curr_fr >= max(cent_on{end})
            ic = ic-1;
            continue
        end
    end
end
clear dlcvideo_meta

refimage = imread('target_0.png');
ref_h = size(refimage,1);
ref_w = size(refimage,2);

% precompute left corner ordering - unchanged across trials
target_left = nan(size(targ_list_left));
xcenter = mean(targ_list_left(:,1));
ycenter = mean(targ_list_right(:,2));
for i = 1:size(targ_list_left,1)
    xdist = targ_list_left(i,1) - xcenter;
    ydist = targ_list_left(i,2) - ycenter;
    if xdist>0 && ydist>0
        target_left(3,:) = targ_list_left(i,:);
    elseif xdist<0 && ydist>0
        target_left(4,:) = targ_list_left(i,:);
    elseif xdist>0 && ydist<0
        target_left(2,:) = targ_list_left(i,:);
    elseif xdist<0 && ydist<0
        target_left(1,:) = targ_list_left(i,:);
    end
end

% precompute right corner ordering - unchanged across trials
target_right = nan(size(targ_list_right));
xcenter = mean(targ_list_right(:,1));
ycenter = mean(targ_list_right(:,2));
for i = 1:length(targ_list_right)
    xdist = targ_list_right(i,1) - xcenter;
    ydist = targ_list_right(i,2) - ycenter;
    if xdist>0 && ydist>0
        target_right(3,:) = targ_list_right(i,:);
    elseif xdist<0 && ydist>0
        target_right(4,:) = targ_list_right(i,:);
    elseif xdist>0 && ydist<0
        target_right(2,:) = targ_list_right(i,:);
    elseif xdist<0 && ydist<0
        target_right(1,:) = targ_list_right(i,:);
    end
end

% precompute center corner ordering - unchanged across trials
target_center = nan(size(targ_list_cent));
xcenter = mean(targ_list_cent(:,1));
ycenter = mean(targ_list_cent(:,2));
for i = 1:length(targ_list_cent)
    xdist = targ_list_cent(i,1) - xcenter;
    ydist = targ_list_cent(i,2) - ycenter;
    if xdist>0 && ydist>0
        target_center(3,:) = targ_list_cent(i,:);
    elseif xdist<0 && ydist>0
        target_center(4,:) = targ_list_cent(i,:);
    elseif xdist>0 && ydist<0
        target_center(2,:) = targ_list_cent(i,:);
    elseif xdist<0 && ydist<0
        target_center(1,:) = targ_list_cent(i,:);
    end
end

% precompute proj_location - depends only on yn_flip and refimage
proj_location = [0 0;ref_w 0;ref_w ref_h;0 ref_h];

sigma = 30;

% load and preprocess all camel images once before the loop
camellist = imageDatastore('camels');
% camellist.Files(449:450) = [];
allcamels = zeros(ref_h, ref_w, length(camellist.Files), 'uint8');
for curr_image = 1:length(camellist.Files)
    temp = imread(camellist.Files{curr_image});
    if size(temp,3) > 1
        temp = rgb2gray(temp);
    end
    allcamels(:,:,curr_image) = imresize(temp, [ref_h ref_w]);
end

% load and preprocess all wrench images once before the loop
wrenchlist = imageDatastore('wrenches');
% wrenchlist.Files(448:450) = [];
allwrenches = zeros(ref_h, ref_w, length(wrenchlist.Files), 'uint8');
for curr_image = 1:length(wrenchlist.Files)
    temp = imread(wrenchlist.Files{curr_image});
    if size(temp,3) > 1
        temp = rgb2gray(temp);
    end
    allwrenches(:,:,curr_image) = imresize(temp, [ref_h ref_w]);
end

% flatten to 2D for batch corr: each column is one image
allcamels_flat  = reshape(single(allcamels),  [], size(allcamels,3));
allwrenches_flat = reshape(single(allwrenches), [], size(allwrenches,3));

% precompute category index vector and preallocate outputs
compImages_ind = [ones(size(allcamels,3),1); ones(size(allwrenches,3),1)*2];
imageTypes = {'camel','wrench'};
camel_side = nan(1, length(side_on));
imageID    = nan(1, length(side_on));

for curr_tr = 1:length(side_on)
    disp(curr_tr)

    % take average over all frames within a trial
    frames_avg = uint8(mean(dlcFrame_side{curr_tr},4));
    % extract the masked part of image & convert to grayscale
    maskedRgbImage = bsxfun(@times, frames_avg, cast(LeftMask, 'like', frames_avg));
    I = rgb2gray(maskedRgbImage);

    % calculate transformation and apply warp to image extracted from video
    transform = fitgeotrans(target_left, proj_location, 'projective');
    Jregistered = imwarp(I, transform, 'OutputView', imref2d(size(I)));
    Jregistered = Jregistered(1:ref_h, 1:ref_w, :);

    % code that removes average background gradient
    I = im2double(Jregistered);
    y = mean(I(:,1:100), 2);
    yb2 = imgaussfilt(y, sigma);
    Ic = bsxfun(@plus, I, yb2(1) - yb2);
    Jregistered = Ic;

    % now load all original camel images and compute pixelwise correlations with extracted frame
    compCImages = corr(single(Jregistered(:)), allcamels_flat);
    compWImages = corr(single(Jregistered(:)), allwrenches_flat);

    allComp = [compCImages compWImages];
    max_idx = find(allComp==max(allComp));
    left_image_type = compImages_ind(max_idx(1));
    left_image_corr = max(allComp);

    if compImages_ind(max_idx(1))==1
        im_idx = max_idx(1);
        filesplit = split(camellist.Files{im_idx},'_');
        left_imageID = str2num(filesplit{end}(1:end-4));
    elseif compImages_ind(max_idx(1))==2
        im_idx = max_idx(1) - size(allcamels,3);
        filesplit = split(wrenchlist.Files{im_idx},'_');
        left_imageID = str2num(filesplit{end}(1:end-4));
    end

    % extract the masked part of image & convert to grayscale
    maskedRgbImage = bsxfun(@times, frames_avg, cast(RightMask, 'like', frames_avg));
    I = rgb2gray(maskedRgbImage);

    % calculate transformation and apply warp to image extracted from video
    transform = fitgeotrans(target_right, proj_location, 'projective');
    Jregistered = imwarp(I, transform, 'OutputView', imref2d(size(I)));
    Jregistered = Jregistered(1:ref_h, 1:ref_w, :);

    % code that removes average background gradient
    I = im2double(Jregistered);
    y = mean(I(:,1:100), 2);
    yb2 = imgaussfilt(y, sigma);
    Ic = bsxfun(@plus, I, yb2(1) - yb2);
    Jregistered = Ic;

    % now load all original camel images and compute pixelwise correlations with extracted frame
    compCImages = corr(single(Jregistered(:)), allcamels_flat);
    compWImages = corr(single(Jregistered(:)), allwrenches_flat);

    allComp = [compCImages compWImages];
    max_idx = find(allComp==max(allComp));
    right_image_type = compImages_ind(max_idx(1));
    right_image_corr = max(allComp);

    if compImages_ind(max_idx(1))==1
        filesplit = split(camellist.Files{max_idx(1)},'_');
        right_imageID = str2num(filesplit{end}(1:end-4));
    elseif compImages_ind(max_idx(1))==2
        im_idx = max_idx(1) - size(allcamels,3);
        filesplit = split(wrenchlist.Files{im_idx},'_');
        right_imageID = str2num(filesplit{end}(1:end-4));
    end

    % if both images are identified as camels, choose the one with greater
    % correlation to be the camel
    if right_image_type==1 && left_image_type==1
        if right_image_corr > left_image_corr
            camel_side(curr_tr) = 2;
            imageID(curr_tr) = right_imageID;
        else
            camel_side(curr_tr) = 1;
            imageID(curr_tr) = left_imageID;
        end
    elseif right_image_type == 1
        camel_side(curr_tr) = 2;
        imageID(curr_tr) = right_imageID;
    elseif left_image_type == 1
        camel_side(curr_tr) = 1;
        imageID(curr_tr) = left_imageID;
    else
        camel_side(curr_tr) = NaN;
        imageID(curr_tr) = NaN;
    end

end

end