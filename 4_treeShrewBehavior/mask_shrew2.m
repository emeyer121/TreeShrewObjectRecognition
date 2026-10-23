function [shrewMask_side, shrewMask_cent, shrewTopPoint_side, shrewTopPoint_cent] = mask_shrew2(dlcFrame_side, dlcFrame_cent)

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

[imgH_excl, imgW_excl] = size(rgb2gray(I));   % full image size

fig_bound = figure();
imagesc(I);
colormap(gray);
title({'Click TWO points (left -> right) to define the upper exclusion boundary.', ...
       'Everything ABOVE this line will be removed from every mask.'});

[bnd_x, bnd_y] = ginput(2);     % two clicks define the line

% Draw the line so the user can confirm it looks right
hold on;
plot(bnd_x, bnd_y, 'r-', 'LineWidth', 2);
title('Boundary line shown in red. Close window to continue.');
waitfor(fig_bound);              % user closes the window to proceed

% Build a logical mask: TRUE = keep, FALSE = exclude (above the line)
exclusionMask = true(imgH_excl, imgW_excl);

x1 = bnd_x(1);  y1 = bnd_y(1);
x2 = bnd_x(2);  y2 = bnd_y(2);

for col = 1:imgW_excl
    % y-value of the boundary line at this column (linear interpolation)
    lineY = y1 + (y2 - y1) * (col - x1) / (x2 - x1);
    rowsToExclude = 1 : max(0, floor(lineY));   % rows above the line
    exclusionMask(rowsToExclude, col) = false;
end

while ~strcmp(targets_good,'Y')

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
    stats_shrew = stats(stats.Area>10,:);

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
    finalMask = finalMask & exclusionMask;

    % get topmost major axis point for visual confirmation
    [topPoint_x, topPoint_y, centroid, majorAxis] = getMajorAxisTop(finalMask);
    
    [imgH_vis, imgW_vis] = size(finalMask);
    L = max(imgH_vis, imgW_vis);                    % line length
    axPt1 = centroid - L * majorAxis';
    axPt2 = centroid + L * majorAxis';

    clf('reset')
    fig3 = figure();
    imshow(labeloverlay(I, finalMask, 'Colormap', [1 0 0], 'Transparency', 0.6));
    hold on;
    plot([axPt1(1) axPt2(1)], [axPt1(2) axPt2(2)], 'c-', 'LineWidth', 1.5);  % axis line
    plot(centroid(1), centroid(2), 'yo', 'MarkerSize', 8, 'LineWidth', 1.5); % centroid
    plot(topPoint_x, topPoint_y, 'g+', 'MarkerSize', 14, 'LineWidth', 2);    % top point
    xlim([0.5 imgW_vis + 0.5]);
    ylim([0.5 imgH_vis + 0.5]);
    title({'Red = mask   Cyan = major axis   Yellow = centroid   Green + = top point'});
    
    targets_good=input('Does the mask look good? Y/N: ','s');
    if ~strcmp(targets_good,'Y')
        fprintf('Redoing target definition \n')
        filtthresh = input(['New filter threshold (previously ',num2str(filtthresh),'): ']);
        close(fig3);
    else
        fprintf('Targets are good. \n')
    end
end

% -- Side view loop -----------------------------------------------------------
shrewMask_side     = nan(size(dlcFrame_side{1},1), size(dlcFrame_side{1},2), length(dlcFrame_side));
shrewTopPoint_side = nan(length(dlcFrame_side), 2);   % [x, y] per frame

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
        finalMask = bwareafilt(masked_regions, 1);
    end
    finalMask = imfill(finalMask, 'holes');
    finalMask = finalMask & exclusionMask;

    [shrewTopPoint_side(i,1), shrewTopPoint_side(i,2)] = getMajorAxisTop(finalMask);
    shrewMask_side(:,:,i) = finalMask;

end

% -- Central view loop --------------------------------------------------------
shrewMask_cent     = nan(size(dlcFrame_cent{1},1), size(dlcFrame_cent{1},2), length(dlcFrame_cent));
shrewTopPoint_cent = nan(length(dlcFrame_cent), 2);   % [x, y] per frame

for i = 1:length(dlcFrame_cent)
    if isempty(dlcFrame_cent{i})
        break
    end

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
        finalMask = bwareafilt(masked_regions, 1);
    end
    finalMask = imfill(finalMask, 'holes');
    finalMask = finalMask & exclusionMask;

    [shrewTopPoint_cent(i,1), shrewTopPoint_cent(i,2)] = getMajorAxisTop(finalMask);
    shrewMask_cent(:,:,i) = finalMask;

end

end   % -- end main function --------------------------------------------------


function [topPoint_x, topPoint_y, centroid, majorAxis] = getMajorAxisTop(finalMask)
% Returns the extreme top boundary point along the major axis.
% The original mask shape is preserved for the caller; cleaning is only
% used internally to stabilize the PCA axis estimate.

    topPoint_x = NaN;  topPoint_y = NaN;
    centroid   = [NaN NaN];
    majorAxis  = [NaN; NaN];

    % NaN ? false
    origMask = finalMask == 1;

    if ~any(origMask(:))
        return;
    end

    % Internal copy for stable axis computation
    % (small opening removes spurs; largest region only)
    axisMask = imopen(origMask, strel('disk', 1));   % smaller radius, safer
    axisMask = bwareafilt(axisMask, 1);

    % If opening wiped the mask out, fall back to raw mask
    if ~any(axisMask(:))
        axisMask = bwareafilt(origMask, 1);
    end

    % PCA on cleaned mask pixels
    [rows, cols] = find(axisMask);
    pts      = [cols, rows];
    centroid = mean(pts, 1);
    centered = pts - centroid;

    [V, ~]    = eig(cov(centered));
    majorAxis = V(:, end);

    % Sample along axis line
    [imgH, imgW] = size(origMask);
    maxExtent    = max(imgH, imgW);
    t  = linspace(-maxExtent, maxExtent, 4*maxExtent);
    xS = centroid(1) + t * majorAxis(1);
    yS = centroid(2) + t * majorAxis(2);

    inB = xS >= 1 & xS <= imgW & yS >= 1 & yS <= imgH;
    xS = xS(inB);  yS = yS(inB);  tInB = t(inB);

    % Check samples against the ORIGINAL mask so returned point is on it
    onMask = origMask(sub2ind([imgH imgW], round(yS), round(xS)));
    if ~any(onMask), return; end

    tOn = tInB(onMask);  xOn = xS(onMask);  yOn = yS(onMask);
    [~, i1] = min(tOn);
    [~, i2] = max(tOn);

    if yOn(i1) <= yOn(i2)
        topPoint_x = xOn(i1);  topPoint_y = yOn(i1);
    else
        topPoint_x = xOn(i2);  topPoint_y = yOn(i2);
    end

    topPoint_x = round(topPoint_x);
    topPoint_y = round(topPoint_y);

    % Safety: ensure the returned pixel is actually on the original mask
    if ~origMask(topPoint_y, topPoint_x)
        [br, bc] = find(bwperim(origMask));
        d = (bc - topPoint_x).^2 + (br - topPoint_y).^2;
        [~, k] = min(d);
        topPoint_x = bc(k);
        topPoint_y = br(k);
    end
end