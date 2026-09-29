function [imageID,camel_side] = correlateStim_autoit(dlcvideo,right_on,left_on,...
    RightMask,LeftMask,refimage,target_list_left,target_list_right,yn_flip)

for curr_tr = 1:length(right_on) % trial number which pulls all frames during image presentation

% take average over all frames within a trial
frames_avg = uint8(mean(dlcvideo(:,:,:,right_on{curr_tr}),4));
% extract the masked part of image & convert to grayscale
maskedRgbImage = bsxfun(@times, frames_avg, cast(LeftMask, 'like', frames_avg));
I=rgb2gray(maskedRgbImage);

% figure;
% imshow(I);hold on;
% plot(target_list_left(:,1),target_list_left(:,2),'o')

% this section determines the order of the stimulus corners based on signs
% of the x & y distances to the center (1: top right, 2: top left, 3:bottom
% left, 4: bottom right)
target_left = nan(size(target_list_left));
xcenter = mean(target_list_left(:,1));
ycenter = mean(target_list_right(:,2));
for i = 1:size(target_list_left,1)
    xdist = target_list_left(i,1) - xcenter;
    ydist = target_list_left(i,2) - ycenter;
    if xdist>0 && ydist>0
        target_left(3,:) = target_list_left(i,:);
    elseif xdist<0 && ydist>0
        target_left(4,:) = target_list_left(i,:);
    elseif xdist>0 && ydist<0
        target_left(2,:) = target_list_left(i,:);
    elseif xdist<0 && ydist<0
        target_left(1,:) = target_list_left(i,:);
    end
end

% location of projected points depending on if video is flipped or not
if strcmp(yn_flip,'N')
    proj_location = [size(refimage,2) 0;0 0;0 size(refimage,1);size(refimage,2) size(refimage,1)];
elseif strcmp(yn_flip,'Y')
    proj_location = [0 0;size(refimage,2) 0;size(refimage,2) size(refimage,1);0 size(refimage,1)];
end

% calculate transformation and apply warp to image extracted from video
transform = fitgeotrans(target_left,proj_location,'projective');
Jregistered = imwarp(I,transform,'OutputView',imref2d(size(I)));

% figure;
% imagesc(Jregistered);colormap('gray');

Jregistered=Jregistered(1:size(refimage,1),1:size(refimage,2),:);

% figure;
% imagesc(Jregistered);colormap('gray');

% code that removes average background gradient
I = im2double(Jregistered);
y = mean(I(:,1:100),2);
sigma = 30;
yb2 = imgaussfilt(y,sigma);
Ic = bsxfun(@plus, I, yb2(1) - yb2);

% figure;
% subplot(1,2,1)
% imshow(Jregistered)
% subplot(1,2,2)
% imshow(Ic)

Jregistered = Ic; % saves Jregistered as new image

% now load all original camel images and compute pixelwise correlations with extracted frame    
camellist=imageDatastore('camels');
camellist.Files(449:450) = [];
for curr_image=1:size(camellist.Files,1)
    temp=imread(camellist.Files{curr_image});
    if size(temp,3)>1
     temp=rgb2gray(temp);
    end
    temp=imresize(temp,[size(refimage,1) size(refimage,2)]);
    allcamels(:,:,curr_image)=temp;
end
for curr_image = 1:size(allcamels,3)
    temp=allcamels(:,:,curr_image);
    compCImages(curr_image)=corr(single(Jregistered(:)),single(temp(:)));
end

% Do the same for wrenches (distractors)
wrenchlist=imageDatastore('wrenches');
wrenchlist.Files(448:450) = [];
for curr_image=1:size(wrenchlist.Files,1)
    temp=imread(wrenchlist.Files{curr_image});
    if size(temp,3)>1
     temp=rgb2gray(temp);
    end
    temp=imresize(temp,[size(refimage,1) size(refimage,2)]);
    allwrenches(:,:,curr_image)=temp;
end
allwrenches = uint8(allwrenches);
for curr_image = 1:size(allwrenches,3)
    temp=allwrenches(:,:,curr_image);
    compWImages(curr_image)=corr(single(Jregistered(:)),single(temp(:)));
end

% Do the same for rhinos (distractors)
rhinolist=imageDatastore('rhinos');
% rhinolist.Files(448:450) = [];
for curr_image=1:size(rhinolist.Files,1)
    temp=imread(rhinolist.Files{curr_image});
    if size(temp,3)>1
     temp=rgb2gray(temp);
    end
    temp=imresize(temp,[size(refimage,1) size(refimage,2)]);
    allrhinos(:,:,curr_image)=temp;
end
allrhinos = uint8(allrhinos);
for curr_image = 1:size(allrhinos,3)
    temp=allrhinos(:,:,curr_image);
    compRImages(curr_image)=corr(single(Jregistered(:)),single(temp(:)));
end

% make a vector with 1s for camel, 2s for wrenches, 3s for rhinos
compImages_ind = [ones(length(compCImages),1); ones(length(compWImages),1)*2; ones(length(compWImages),1)*3];
% append all correlation values for each stimulus
allComp = [compCImages compWImages compRImages];
% find the max correlation and determine which category it comes from
max_idx = find(allComp==max(allComp));
imageTypes = {'camel','wrench','rhino'};
left_image_type = compImages_ind(max_idx(1));
left_image_corr = max(allComp);

% based on which category it comes from, save the image ID
if compImages_ind(max_idx(1))==1
    im_idx = max_idx(1);
    filesplit = split(camellist.Files{im_idx},'_');
    left_imageID = str2num(filesplit{end}(1:end-4));
elseif compImages_ind(max_idx(1))==2
    im_idx = max_idx(1) - length(compCImages);
    filesplit = split(wrenchlist.Files{im_idx},'_');
    left_imageID = str2num(filesplit{end}(1:end-4));
else
    im_idx = max_idx(1) - length(compCImages) - length(compWImages);
    filesplit = split(rhinolist.Files{im_idx},'_');
    left_imageID = str2num(filesplit{end}(1:end-4));
end

% plots image (left) with closest camel, wrench, and rhino

% if curr_tr ==4
% figure;
% subplot(1,4,1);
% imshow(Jregistered);
% title(['Image is: ', imageTypes{compImages_ind(allComp==max(allComp))}, ' ', num2str(left_imageID), ', Corr = ', num2str(max(allComp))])
% subplot(1,4,2);
% cind = find(compCImages==max(compCImages));
% imshow(allcamels(:,:,cind(1))); colormap(gray)
% subplot(1,4,3);
% wind = find(compWImages==max(compWImages));
% imshow(allwrenches(:,:,wind(1))); colormap(gray)
% subplot(1,4,4);
% rind = find(compRImages==max(compRImages));
% imshow(allrhinos(:,:,rind(1))); colormap(gray)
% end

% take average over all frames within a trial
frames_avg = uint8(mean(dlcvideo(:,:,:,left_on{curr_tr}),4));
% extract the masked part of image & convert to grayscale
maskedRgbImage = bsxfun(@times, frames_avg, cast(RightMask, 'like', frames_avg));
I=rgb2gray(maskedRgbImage);

% this section determines the order of the stimulus corners based on signs
% of the x & y distances to the center (1: top right, 2: top left, 3:bottom
% left, 4: bottom right)
target_right = nan(size(target_list_right));
xcenter = mean(target_list_right(:,1));
ycenter = mean(target_list_right(:,2));
for i = 1:length(target_list_right)
    xdist = target_list_right(i,1) - xcenter;
    ydist = target_list_right(i,2) - ycenter;
    if xdist>0 && ydist>0
        target_right(3,:) = target_list_right(i,:);
    elseif xdist<0 && ydist>0
        target_right(4,:) = target_list_right(i,:);
    elseif xdist>0 && ydist<0
        target_right(2,:) = target_list_right(i,:);
    elseif xdist<0 && ydist<0
        target_right(1,:) = target_list_right(i,:);
    end
end

% location of projected points depending on if video is flipped or not
if strcmp(yn_flip,'N')
    proj_location = [size(refimage,2) 0;0 0;0 size(refimage,1);size(refimage,2) size(refimage,1)];
elseif strcmp(yn_flip,'Y')
    proj_location = [0 0;size(refimage,2) 0;size(refimage,2) size(refimage,1);0 size(refimage,1)];
end

% calculate transformation and apply warp to image extracted from video 
transform = fitgeotrans(target_right,proj_location,'projective');
Jregistered = imwarp(I,transform,'OutputView',imref2d(size(I)));

% figure;
% imagesc(Jregistered);colormap('gray');

Jregistered=Jregistered(1:size(refimage,1),1:size(refimage,2),:);

% figure;
% imagesc(Jregistered);colormap('gray');

% code that removes average background gradient
I = im2double(Jregistered);
y = mean(I(:,1:100),2);
sigma = 30;
yb2 = imgaussfilt(y,sigma);
Ic = bsxfun(@plus, I, yb2(1) - yb2);

% figure;
% subplot(1,2,1)
% imshow(Jregistered)
% subplot(1,2,2)
% imshow(Ic)
Jregistered = Ic; % saves Jregistered as new image

% now load all original camel images and compute pixelwise correlations with extracted frame    
camellist=imageDatastore('camels');
camellist.Files(449:450) = [];
for curr_image=1:size(camellist.Files,1)
    temp=imread(camellist.Files{curr_image});
    if size(temp,3)>1
     temp=rgb2gray(temp);
    end
    temp=imresize(temp,[size(refimage,1) size(refimage,2)]);
    allcamels(:,:,curr_image)=temp;
end
for curr_image = 1:size(allcamels,3)
    temp=allcamels(:,:,curr_image);
    compCImages(curr_image)=corr(single(Jregistered(:)),single(temp(:)));
end

% Do the same for wrenches (distractors)
wrenchlist=imageDatastore('wrenches');
wrenchlist.Files(448:450) = [];
for curr_image=1:size(wrenchlist.Files,1)
    temp=imread(wrenchlist.Files{curr_image});
    if size(temp,3)>1
     temp=rgb2gray(temp);
    end
    temp=imresize(temp,[size(refimage,1) size(refimage,2)]);
    allwrenches(:,:,curr_image)=temp;
end
allwrenches = uint8(allwrenches);
for curr_image = 1:size(allwrenches,3)
    temp=allwrenches(:,:,curr_image);
    compWImages(curr_image)=corr(single(Jregistered(:)),single(temp(:)));
end

% Do the same for rhinos (distractors)
rhinolist=imageDatastore('rhinos');
% rhinolist.Files(448:450) = [];
for curr_image=1:size(rhinolist.Files,1)
    temp=imread(rhinolist.Files{curr_image});
    if size(temp,3)>1
     temp=rgb2gray(temp);
    end
    temp=imresize(temp,[size(refimage,1) size(refimage,2)]);
    allrhinos(:,:,curr_image)=temp;
end
allrhinos = uint8(allrhinos);
for curr_image = 1:size(allrhinos,3)
    temp=allrhinos(:,:,curr_image);
    compRImages(curr_image)=corr(single(Jregistered(:)),single(temp(:)));
end

% make a vector with 1s for camel, 2s for wrenches, 3s for rhinos
compImages_ind = [ones(length(compCImages),1); ones(length(compWImages),1)*2; ones(length(compWImages),1)*3];
% append all correlation values for each stimulus
allComp = [compCImages compWImages compRImages];
% find the max correlation and determine which category it comes from
max_idx = find(allComp==max(allComp));
imageTypes = {'camel','wrench','rhino'};
right_image_type = compImages_ind(max_idx(1));
right_image_corr = max(allComp);
% based on which category it comes from, save the image ID
if compImages_ind(max_idx(1))==1
    filesplit = split(camellist.Files{max_idx(1)},'_');
    right_imageID = str2num(filesplit{end}(1:end-4));
elseif compImages_ind(max_idx(1))==2
    im_idx = max_idx(1) - length(compCImages);
    filesplit = split(wrenchlist.Files{im_idx},'_');
    right_imageID = str2num(filesplit{end}(1:end-4));
else
    im_idx = max_idx(1) - length(compCImages) - length(compWImages);
    filesplit = split(rhinolist.Files{im_idx},'_');
    right_imageID = str2num(filesplit{end}(1:end-4));
end


% plots image (left) with closest camel, wrench, and rhino

% ** uncomment for plotting **
% if curr_tr==4
% figure;
% subplot(1,4,1);
% imshow(Jregistered);
% title(['Image is: ', imageTypes{compImages_ind(allComp==max(allComp))}, ' ', num2str(right_imageID), ', Corr = ', num2str(max(allComp))])
% subplot(1,4,2);
% cind = find(compCImages==max(compCImages));
% imshow(allcamels(:,:,cind(1))); colormap(gray)
% disp(max(compCImages))
% subplot(1,4,3);
% wind = find(compWImages==max(compWImages));
% imshow(allwrenches(:,:,wind(1))); colormap(gray)
% disp(max(compWImages))
% subplot(1,4,4);
% rind = find(compRImages==max(compRImages));
% imshow(allrhinos(:,:,rind(1))); colormap(gray)
% disp(max(compRImages))
% end
% **

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
% if the right side is the camel,...
elseif right_image_type == 1
    camel_side(curr_tr) = 2;
    imageID(curr_tr) = right_imageID;
% if the left side is the camel,...
elseif left_image_type == 1
    camel_side(curr_tr) = 1;
    imageID(curr_tr) = left_imageID;
% if neither side is identified as a camel, make camel_side and imageID=NaN
else
    camel_side(curr_tr) = NaN;
    imageID(curr_tr) = NaN;
end

end