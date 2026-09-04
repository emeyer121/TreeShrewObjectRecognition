function [nullAcc, pval] = chanceNullBySession(nTrialsPerSess, obsAcc, nBoot, chanceLevel)
%CHANCENULLBYSESSION  Simulated null distribution: pooled accuracy under
%   pure chance guessing, using the same per-session trial counts as your
%   real data (so session clustering matches the real analysis).
%
%   nTrialsPerSess : vector, trial count per session, matched to whatever
%                    you're testing, e.g.:
%                      cellfun(@length, sessionPerf_nonCatch{task,ts})
%   obsAcc         : observed pooled accuracy, for the p-value
%   nBoot          : number of simulated datasets (default 10000)
%   chanceLevel    : guess rate to simulate (default 0.5)
%
%   nullAcc : nBoot x 1, simulated pooled accuracy under chance guessing
%   pval    : P(simulated accuracy >= obsAcc | pure chance)

if nargin < 3 || isempty(nBoot), nBoot = 10000; end
if nargin < 4 || isempty(chanceLevel), chanceLevel = 0.5; end

nTrialsPerSess = nTrialsPerSess(nTrialsPerSess > 0);
nullAcc = nan(nBoot,1);
for b = 1:nBoot
    simTrials = arrayfun(@(n) rand(n,1) < chanceLevel, nTrialsPerSess, 'UniformOutput', false);
    nullAcc(b) = mean(vertcat(simTrials{:}));
end

pval = mean(nullAcc >= obsAcc);
end