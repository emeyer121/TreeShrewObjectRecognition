function [side_on, cent_on] = define_trials(vidPath,RightMask,LeftMask,CentMask,stim_thresh)

dlcvideo_meta = VideoReader(vidPath);
max_frame = dlcvideo_meta.NumFrames;
for curr_tr=1:max_frame
    dlcFrame = readFrame(dlcvideo_meta);
    maskedRgbImage = bsxfun(@times, dlcFrame, cast(RightMask, 'like', dlcFrame));
    rightside_intensity(curr_tr)=mean(maskedRgbImage(:));
    maskedRgbImage = bsxfun(@times, dlcFrame, cast(LeftMask, 'like', dlcFrame));
    leftside_intensity(curr_tr)=mean(maskedRgbImage(:));
    maskedRgbImage = bsxfun(@times, dlcFrame, cast(CentMask, 'like', dlcFrame));
    center_intensity(curr_tr)=mean(maskedRgbImage(:));
end
clear dlcvideo_meta

% want to find periods of time where there are no trials within 100 frames
% so we can normalize each chunk of trials separately
thresh = find(abs(diff(rightside_intensity))>0.1);
findgaps = [0 thresh(diff(thresh)>100)];
len = length(rightside_intensity);

% for each chunk of trials we subtract the mean intensity during that
% period
for i = 1:length(findgaps)
    if i==length(findgaps)
        inter = findgaps(i)+1:len;
    else
        inter = findgaps(i)+1:findgaps(i+1);
    end
    rightside_intensity(inter) = rightside_intensity(inter) - min(rightside_intensity(inter));
    leftside_intensity(inter) = leftside_intensity(inter) - min(leftside_intensity(inter));
    center_intensity(inter) = center_intensity(inter) - min(center_intensity(inter));
end


% plot the intensities post-normalization

figure();
plot(rightside_intensity)
hold on;
plot(leftside_intensity)
plot(center_intensity)
yline(stim_thresh,'k','linewidth',2)
xlabel('Frame Number','FontSize',15)
ylabel('Stimulus Intensity','FontSize',15)
legend('Right Stim.','Left Stim.','Cent. Stim')

% find frames for each image where the intensity < int_th (defined in main
% script) and save in new cell for right/left stims
rightside_on = find(rightside_intensity<stim_thresh);
right_diff = sum(diff(rightside_on)>1)+1;
right_on = cell(1,right_diff); k = 1;
for i = 1:length(rightside_on)-1
    right_on{k} = [right_on{k} rightside_on(i)];
    if rightside_on(i+1)-rightside_on(i)>1
        k = k+1;
    end
end
leftside_on = find(leftside_intensity<stim_thresh);
left_diff = sum(diff(leftside_on)>1)+1;
left_on = cell(1,left_diff); k = 1;
for i = 1:length(leftside_on)-1
    left_on{k} = [left_on{k} leftside_on(i)];
    if leftside_on(i+1)-leftside_on(i)>1
        k = k+1;
    end
end

center_on = find(center_intensity<stim_thresh);
cent_diff = sum(diff(center_on)>1)+1;
cent_on = cell(1,cent_diff); k = 1;
for i = 1:length(center_on)-1
    cent_on{k} = [cent_on{k} center_on(i)];
    if center_on(i+1)-center_on(i)>1
        k = k+1;
    end
end

% don't really need both right_on and left_on because if they are different
% lengths I set them equal to each other, but helps to keep track of the side
if length(left_on)~=length(right_on)
    y = find(min([length(left_on) length(right_on)]));
    if y==1
        right_on = left_on;
    else
        left_on = right_on;
    end
end
side_on = right_on;

% if any trial lengths last 1 frame or have empty cells, I remove them -
% not sure if this actually happens, but it's a precaution
side_on(cellfun(@(x) length(x), side_on)>100 | cellfun(@(x) length(x), side_on)<2) = [];

end