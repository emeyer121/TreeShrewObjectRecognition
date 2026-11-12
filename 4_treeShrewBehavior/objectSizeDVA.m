%% objectSizeDVA.m
% Size in degrees of visual angle (dva) of every target and distractor image
% shown in three behavior tasks:
%   Camel_v2_test_nn        targets: camels           distractors: wrenches
%   Camel_novel_v2_test_nn  targets: novel camels (ID > 400) + familiar
%                           Camel_v2 camels (ID <= 200)
%                           distractors: novel distractors (ID > 400) +
%                           familiar Camel_v2 wrenches (ID <= 200)
%   Camel_Rhino_test_nn     targets: camels (0-400)   distractors: rhinos
% Image IDs come from T_Expt_ID / D_Expt_ID in behaviorData/<task>.csv and
% images from stimulusSets/<task>/original/.
%
% Each image spans 30 dva (dva per pixel = 30 / image width, so 1200 px
% images are handled too). The object mask is every pixel more than 1.5 gray
% levels (8-bit units) from the background, keeping only the largest
% 8-connected piece. This removes faint paste-outline rectangles and specks
% present in some novel images; for the clean images it is the whole object.
%
% Size is the tightest bounding box at any rotation (minimum-width bounding
% rectangle), measured to pixel edges:
%   minDVA  - short side of the box = narrowest width of the object
%   maxDVA  - long side of the same box (perpendicular to minDVA)
%   meanDVA - average of the two
% The short side is cross-checked against regionprops' MinFeretDiameter.
% Objects that touch the image edge are included and flagged.

scriptDir   = fileparts(mfilename('fullpath'));
behavDir    = fullfile(scriptDir, 'behaviorData');
stimDir     = fullfile(scriptDir, '..', 'stimulusSets');
outFile     = fullfile(behavDir, 'objectSizeDVA.mat');

imageFOVdeg = 30;        % visual angle spanned by the full image
bgTol       = 1.5;       % gray levels (8-bit) from background still counted as background
                         % (so 126-128 = background; 16-bit 32255/256 = 125.996)

tasks = {'Camel_v2_test_nn', 'Camel_novel_v2_test_nn', 'Camel_Rhino_test_nn'};
roles = {'target', 'distractor'};
idCol = struct('target', 'T_Expt_ID', 'distractor', 'D_Expt_ID');

perImage = table();
for t = 1:numel(tasks)
    task = tasks{t};
    T = readtable(fullfile(behavDir, [task '.csv']), 'VariableNamingRule', 'preserve');
    for r = 1:numel(roles)
        role = roles{r};
        ids = unique(T.(idCol.(role)));
        ids = ids(~isnan(ids));
        for i = 1:numel(ids)
            [file, source] = imageFile(stimDir, task, role, ids(i));
            m = measureObject(file, imageFOVdeg, bgTol);
            row = table(string(task), string(role), ids(i), string(source), ...
                m.widthPx, m.shortPx, m.longPx, m.angle, m.minDVA, m.maxDVA, m.meanDVA, ...
                m.touchesEdge, m.droppedPx, string(file), ...
                'VariableNames', {'task','role','imageID','source','imageWidthPx', ...
                'shortSidePx','longSidePx','boxAngleDeg','minDVA','maxDVA','meanDVA', ...
                'touchesEdge','droppedPx','file'});
            perImage = [perImage; row]; %#ok<AGROW>
        end
    end
end

%% Summary: min / max / median of each metric, per task and role
metrics = {'minDVA','maxDVA','meanDVA'};
summary = table();
for t = 1:numel(tasks)
    for r = 1:numel(roles)
        sel = perImage.task == tasks{t} & perImage.role == roles{r};
        for k = 1:numel(metrics)
            v = perImage.(metrics{k})(sel);
            summary = [summary; table(string(tasks{t}), string(roles{r}), string(metrics{k}), ...
                nnz(sel), min(v), max(v), median(v), ...
                'VariableNames', {'task','role','metric','nImages','min','max','median'})]; %#ok<AGROW>
        end
    end
end

%% Print
for t = 1:numel(tasks)
    for r = 1:numel(roles)
        sel = perImage.task == tasks{t} & perImage.role == roles{r};
        fprintf('\n===== %s - %ss (%d images, %d touch edge) =====\n', tasks{t}, roles{r}, ...
            nnz(sel), nnz(perImage.touchesEdge(sel)));
        disp(perImage(sel, {'imageID','source','shortSidePx','longSidePx','minDVA','maxDVA','meanDVA','touchesEdge','droppedPx'}))
    end
end
fprintf('\n===== Summary (dva) =====\n');
disp(summary)

save(outFile, 'perImage', 'summary', 'imageFOVdeg', 'bgTol');
fprintf('Saved %s\n', outFile);

%% ------------------------------------------------------------------------
function [file, source] = imageFile(stimDir, task, role, id)
% Map a behavior image ID to its stimulus file.
switch task
    case 'Camel_v2_test_nn'
        source = 'Camel_v2_test_nn';
        pre = struct('target','camel', 'distractor','wrench');
    case 'Camel_novel_v2_test_nn'
        if id > 400                        % novel objects
            source = 'Camel_novel_v2_test_nn';
            pre = struct('target','target', 'distractor','distractor');
        else                               % familiar Camel_v2 images
            source = 'Camel_v2_test_nn';
            pre = struct('target','camel', 'distractor','wrench');
        end
    case 'Camel_Rhino_test_nn'
        source = 'Camel_Rhino_test_nn';
        pre = struct('target','camel', 'distractor','rhino');
end
file = fullfile(stimDir, source, 'original', sprintf('%s_%d.png', pre.(role), id));
if ~exist(file, 'file'), error('Missing image %s', file); end
end

function m = measureObject(file, imageFOVdeg, bgTol)
im = imread(file);                         % alpha channel (if any) is not returned
if isa(im, 'uint16')
    g = double(im) / 256;                  % 16-bit images store value*256 (bg 32512 = 127)
else
    g = double(im);
end
if size(g,3) == 3, g = 0.2989*g(:,:,1) + 0.5870*g(:,:,2) + 0.1140*g(:,:,3); end
if size(g,1) ~= size(g,2), error('%s is not square', file); end

border = [g(1,:), g(end,:), g(:,1)', g(:,end)'];
bg = median(border);
raw = abs(g - bg) > bgTol;
mask = bwareafilt(raw, 1, 8);              % largest 8-connected piece

m.widthPx = size(g,2);
m.droppedPx = nnz(raw) - nnz(mask);
m.touchesEdge = any(mask([1 end],:), 'all') || any(mask(:,[1 end]), 'all');
[m.shortPx, m.longPx, m.angle] = minWidthBox(mask);

s = regionprops(uint8(mask), 'MinFeretProperties');
if abs(s.MinFeretDiameter - m.shortPx) > 1e-6
    error('%s: box short side %.4f px ~= min Feret %.4f px', file, m.shortPx, s.MinFeretDiameter);
end

degPerPx = imageFOVdeg / m.widthPx;
m.minDVA  = m.shortPx * degPerPx;
m.maxDVA  = m.longPx  * degPerPx;
m.meanDVA = (m.minDVA + m.maxDVA) / 2;
end

function [shortSide, longSide, angleDeg] = minWidthBox(mask)
% Minimum-width bounding rectangle of all true pixels in mask (in px).
% Uses the corners of each pixel, so the box encloses whole pixels. The
% narrowest box always has a side along a convex-hull edge, so every hull
% edge is tested. angleDeg is the direction of the long side (x axis, y down).
[r, c] = find(mask);
pts = [c-0.5 r-0.5; c+0.5 r-0.5; c-0.5 r+0.5; c+0.5 r+0.5];
pts = unique(pts, 'rows');
k = convhull(pts(:,1), pts(:,2));
hull = pts(k,:);

shortSide = inf; longSide = NaN; angleDeg = NaN;
for e = 1:size(hull,1)-1
    d = hull(e+1,:) - hull(e,:);
    if ~any(d), continue; end
    u = d / norm(d);
    n = [-u(2) u(1)];
    pn = hull * n';
    width = max(pn) - min(pn);
    if width < shortSide
        pu = hull * u';
        shortSide = width;
        longSide  = max(pu) - min(pu);
        angleDeg  = atan2d(u(2), u(1));
    end
end
end
