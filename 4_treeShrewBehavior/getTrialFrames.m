function [right_on,left_on,leftside_intensity,rightside_intensity] = getTrialFrames(dlcvideo,RightMask,LeftMask,int_th)

% iterate over all frames and extract the intensity changes over the
% right/left stimulus masks
for curr_tr=1:size(dlcvideo,4)
    maskedRgbImage = bsxfun(@times, dlcvideo(:,:,:,curr_tr), cast(RightMask, 'like', dlcvideo(:,:,:,curr_tr)));
    rightside_intensity(curr_tr)=mean(maskedRgbImage(:));
    maskedRgbImage = bsxfun(@times, dlcvideo(:,:,:,curr_tr), cast(LeftMask, 'like', dlcvideo(:,:,:,curr_tr)));
    leftside_intensity(curr_tr)=mean(maskedRgbImage(:));
end

% plot the intensities pre-normalization
% figure();
% plot(rightside_intensity)
% hold on;
% plot(leftside_intensity)

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
    rightside_intensity(inter) = rightside_intensity(inter) - mean(rightside_intensity(inter));
    leftside_intensity(inter) = leftside_intensity(inter) - mean(leftside_intensity(inter));
end

% plot the intensities post-normalization
figure();
plot(rightside_intensity)
hold on;
plot(leftside_intensity)
yline(int_th,'k','linewidth',2)
xlabel('Frame Number','FontSize',15)
ylabel('Stimulus Intensity','FontSize',15)
legend('Right Stim.','Left Stim.')

% find frames for each image where the intensity < int_th (defined in main
% script) and save in new cell for right/left stims
rightside_on = find(rightside_intensity<int_th);
right_diff = sum(diff(rightside_on)>1)+1;
right_on = cell(1,right_diff); k = 1;
for i = 1:length(rightside_on)-1
    right_on{k} = [right_on{k} rightside_on(i)];
    if rightside_on(i+1)-rightside_on(i)>1
        k = k+1;
    end
end
leftside_on = find(leftside_intensity<int_th);
left_diff = sum(diff(leftside_on)>1)+1;
left_on = cell(1,left_diff); k = 1;
for i = 1:length(leftside_on)-1
    left_on{k} = [left_on{k} leftside_on(i)];
    if leftside_on(i+1)-leftside_on(i)>1
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

% if any trial lengths last 1 frame or have empty cells, I remove them -
% not sure if this actually happens, but it's a precaution
trlen = find(cellfun(@(x) length(x),right_on)>100 | cellfun(@(x) length(x),right_on)<2);
right_on(trlen) = [];
left_on(trlen) = [];


end
