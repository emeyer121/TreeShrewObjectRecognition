clear all;clc;

%% Load or specify category labels
addpath(genpath('../helperFunctions/'))
allTasks = {'Camel_v2_test_nn'};%,'Camel_novel_v2_test_nn','Camel_Rhino_test_nn','Camel_background_matrix'};

d = natsortfiles(dir('../4_treeShrewBehavior/behaviorData/shrewData.mat'));
load([d(end).folder,'/',d(end).name])

exp_colors = flipud([179,128,206;...
    211,107,172;...
    254,153,132;...
    248,214,86]/256);

allData.Camel_v2_test_nn.TestTarg(allData.Camel_v2_test_nn.TestTarg==398) = 198;

% Determine which shrews to use
shrewname={'Seymour','Dominic','Ryker'};
shrewID = 1:3;

targdistPerf = cell(length(allTasks),length(shrewname));
targdistPerf_all = cell(length(allTasks),1);
targID = cell(length(allTasks),1);
distID = cell(length(allTasks),1);
ntrials = cell(length(allTasks),length(shrewname));

for task = 1:length(allTasks)
    nTarg = length(unique(allData.(allTasks{task}).T_Expt_ID));
    nDist = length(unique(allData.(allTasks{task}).D_Expt_ID));

    targID{task} = unique(allData.(allTasks{task}).T_Expt_ID);
    distID{task} = unique(allData.(allTasks{task}).D_Expt_ID);
    for ts = 1:length(shrewname)
        for tt = 1:nTarg
            targdistPerf{task,ts}(tt,:) = nan(1,nDist);
            targdistPerf_all{task}(tt,:) = nan(1,nDist);
            ntrials{task,ts}(tt,:) = nan(1,nDist);
            for dd = 1:nDist
                ntrials{task,ts}(tt,dd) = sum(allData.(allTasks{task}).ShrewID==shrewID(ts) & allData.(allTasks{task}).T_Expt_ID==targID{task}(tt) & allData.(allTasks{task}).D_Expt_ID==distID{task}(dd));
                if ntrials{task,ts}(tt,dd) >= 10
                    inclusion = allData.(allTasks{task}).ShrewID==shrewID(ts) & allData.(allTasks{task}).T_Expt_ID==targID{task}(tt) & allData.(allTasks{task}).D_Expt_ID==distID{task}(dd);
                    inclusion_all = allData.(allTasks{task}).T_Expt_ID==targID{task}(tt) & allData.(allTasks{task}).D_Expt_ID==distID{task}(dd);
                    targdistPerf{task,ts}(tt,dd) = mean(allData.(allTasks{task}).correct(inclusion),'omitnan');
                    targdistPerf_all{task}(tt,dd) = mean(allData.(allTasks{task}).correct(inclusion),'omitnan');
                end
            end
        end
    end
end



%%
taskID = 1;

targ_vals = [0  6 9 13 17 32 38 40 44 55 60 65];
dist_vals = [0 14 26 83 87 92 93];

targ_idx = [];
for i = 1:length(targID{taskID})
    if any(targ_vals == targID{taskID}(i))
        targ_idx = [targ_idx, i];
    end
end

dist_idx = [];
for i = 1:length(distID{taskID})
    if any(dist_vals == distID{taskID}(i))
        dist_idx = [dist_idx, i];
    end
end

perf_array = targdistPerf_all{taskID}(targ_idx,dist_idx);

dist_data = load(['../stimulusSets/',allTasks{task},'/wrench_trans_data.mat']);
targ_data = load(['../stimulusSets/',allTasks{task},'/camel_trans_data.mat']);

row0 = {'c000',0,0,0,0,0,1};
targ_data.tab = [row0; targ_data.tab];
row0 = {'w000',0,0,0,0,0,1};
dist_data.tab = [row0; dist_data.tab];

targ_scale = targ_data.tab(targ_vals+1,'Scale');
dist_scale = dist_data.tab(dist_vals+1,'Scale');

[targ_sort, targ_sort_idx] = sort(table2array(targ_scale),'descend');
[dist_sort, dist_sort_idx] = sort(table2array(dist_scale),'descend');

perf_array_sort = perf_array(targ_sort_idx,dist_sort_idx);
perf_array_targ = mean(perf_array_sort,2);
perf_array_dist = mean(perf_array_sort,1);

figure();
ax1 = subplot(2,2,1);
imagesc(perf_array_targ)
pos1 = get(ax1, 'Position');
new_width = pos1(3) * 0.25; % Increase width by 20%
new_x = pos1(1) * 3;

set(ax1, 'Position', [new_x, pos1(2), new_width, pos1(4)]);
yticks(1:12)
yticklabels(round(targ_sort,2))
xticks(1)
xticklabels({'Avg Perf'})
xlabel(' ')
clim([0.5 1])

subplot(2,2,2)
imagesc(perf_array_sort)
yticks(1:12)
yticklabels(round(targ_sort,2))
xticks(1:7)
xticklabels(round(dist_sort,2))
colorbar; clim([0.5 1])
ylabel('Targets')
xlabel('Distractors')

ax2 = subplot(2,2,4);
imagesc(perf_array_dist)
pos1 = get(ax2, 'Position');
new_height = pos1(4) * 0.25;
new_y = pos1(2) * 3;

set(ax2, 'Position', [pos1(1), new_y, pos1(3), new_height]);
yticks(1)
yticklabels({'Avg Perf'})
xticks(1:7)
xticklabels(round(dist_sort,2))
xlabel(' ')
clim([0.5 1])

figure();
size_diff = targ_sort - dist_sort';
size_diff2 = size_diff';
imagesc(size_diff)
yticks(1:12)
yticklabels(targ_sort)
xticks(1:7)
xticklabels(dist_sort)
colorbar;

%%
size_diff = table2array(targ_scale) - table2array(dist_scale)';
size_diff2 = size_diff';
writematrix(size_diff2(:),'./distData/Camel_v2_test_nn/size_differences.csv')