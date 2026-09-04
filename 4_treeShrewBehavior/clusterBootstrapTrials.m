function [obsAcc, ci, pval, bootAcc] = clusterBootstrapTrials(sessionPerf, nBoot, chanceLevel)
%CLUSTERBOOTSTRAPTRIALS  Bootstrap accuracy by resampling sessions, not trials.
%   Trials within a session aren't independent (the same images repeat
%   across sessions), so resampling whole sessions - not individual
%   trials - keeps that structure intact.
%
%   sessionTrials : cell array, one cell per session, each a vector of
%                   0/1 correct/incorrect outcomes for that session
%   nBoot         : number of resamples (default 10000)
%   chanceLevel   : value to test against (default 0.5)
%
%   obsAcc  : observed accuracy, pooling all real trials
%   ci      : [lower upper], 95% percentile bootstrap CI
%   pval    : proportion of bootstrap resamples at/below chanceLevel -
%             a one-sided test of "is accuracy above chance"
%   bootAcc : the full bootstrap distribution (for plotting, optional)

if nargin < 2 || isempty(nBoot), nBoot = 10000; end
if nargin < 3 || isempty(chanceLevel), chanceLevel = 0.5; end

for i = 1:numel(sessionPerf)
    v = sessionPerf{i}(:);
    sessionPerf{i} = v(~isnan(v));
end
sessionPerf = sessionPerf(~cellfun(@isempty, sessionPerf));
nSess = numel(sessionPerf);

obsAcc = mean(vertcat(sessionPerf{:}));

bootAcc = nan(nBoot,1);
for b = 1:nBoot
    idx = randi(nSess, nSess, 1);
    bootAcc(b) = mean(vertcat(sessionPerf{idx}));
end

ci = prctile(bootAcc, [2.5 97.5]);
pval = mean(bootAcc <= chanceLevel);
end