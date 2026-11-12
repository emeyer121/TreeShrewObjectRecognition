%% bench_alloc.m  -- memory-allocation benchmark for forwardRender's compute()
% Runs the exact per-pixel compute() used by forwardRender on block 1
% (Ecc [-13.07, 13.07], 30 deg, 55 px) and measures, per worker:
%   wall time per call, user vs kernel CPU time, page faults, huge pages.
% Called from bench_alloc.sbatch with the variable `cond` already set.
% Does not save anything to the render matrix folders.

if ~exist('cond','var'), cond = 'unnamed'; end
benchDir = pwd;
clearvars -except cond benchDir
cd /mnt/DataDrive3/emeyer/TreeShrewObjRec/isetbio_analyses/
startup
tbUseProject('ISETTreeShrew');

%% Same setup as create_render_matrix_piecewise.m, block (1,1)
dataSavePath = '/mnt/DataDrive2/treeshrew/data_raw/treeshrew_isetbio/';
imOrigSize = 256; imBorder = 11; nBlock = 8; sceneFOVdegs = 30;
blockLen   = imOrigSize/nBlock;
blockSize1 = blockLen + 2*imBorder + 1;
pixDeg     = sceneFOVdegs / imOrigSize;
imCenter   = imBorder + (imOrigSize+1)/2;
col_val = 1:blockSize1; row_val = 1:blockSize1;
eccX =  (mean(col_val) - imCenter) * pixDeg;
eccY = -(mean(row_val) - imCenter) * pixDeg;
mosaicFOVdegs = [length(col_val) length(row_val)]*pixDeg;
blockSize = [blockSize1, blockSize1, 3];
FOV_str = num2str(sceneFOVdegs);
eccX_str = num2str(round(eccX,2)); eccY_str = num2str(round(eccY,2));

retina = ConeResponseCmosaic(eccX,eccY,'fovealDegree',mosaicFOVdegs(1));
load(sprintf('%s/coneMosaics_test/treeshrew_blocked/coneMosaic_deg%s_Xecc%s_Yecc%s_quad.mat', ...
    dataSavePath,FOV_str,eccX_str,eccY_str),'TSConeMosaic')
retina.Mosaic = TSConeMosaic;
retina.PSF = oiTreeShrewCreate('opticsType','wvf','name','wvf-based optics');
retina.psfData = [];
retina.compute(rand(blockSize));      % same warm-up as forwardRender
retina.LastOI = []; retina.LastResponse = [];

%% Benchmark: 1 worker alone, then 4 workers at once
nCalls = 8;                            % timed calls per worker
% /proc/self/stat fields after "(comm) ": 8=minflt, 12=utime, 13=stime (ticks of 1/100 s)
statVec = @() str2double(subsref(strsplit(strtrim(regexprep(fileread('/proc/self/stat'), '^.*\) ', ''))), ...
                                 struct('type','()','subs',{{[8 12 13]}})));
thpMB   = @() str2double(regexp(fileread('/proc/self/smaps_rollup'), 'AnonHugePages:\s+(\d+)', 'tokens', 'once'))/1024;
for nW = [1 4]
    delete(gcp('nocreate'));
    c = parcluster('Processes');
    c.JobStorageLocation = fullfile(getenv('MATLAB_JOBDIR'), sprintf('%s_%d', cond, nW));
    if ~exist(c.JobStorageLocation,'dir'), mkdir(c.JobStorageLocation); end
    c.NumWorkers = nW;
    parpool(c, nW);

    spmd
        r = retina;                    % local copy, like parfor broadcast
        pix = (spmdIndex-1)*100 + (1:nCalls+1);
        in = zeros(blockSize); in(pix(1)) = 1;
        r.compute(in, 'lowOpticalImageResolutionWarning', false);   % untimed warm-up (fills OTF cache)
        s0 = statVec(); t0 = tic;
        for k = 2:nCalls+1
            in = zeros(blockSize); in(pix(k)) = 1;
            r.compute(in, 'lowOpticalImageResolutionWarning', false);
        end
        wall = toc(t0); s1 = statVec();
        d = s1 - s0;                   % [minflt, utime, stime]
        hwm = regexp(fileread('/proc/self/status'), 'VmHWM:\s+(\d+)', 'tokens', 'once');
        res = [wall/nCalls, d(2)/100/nCalls, d(3)/100/nCalls, d(1)/nCalls, thpMB(), str2double(hwm{1})/1024^2];
    end
    R = cell2mat(res(:));
    fprintf('\n=== cond=%s  workers=%d  (per call, mean over workers) ===\n', cond, nW);
    fprintf('  wall %.1f s | user %.1f s | kernel %.1f s | page faults %.2fM | huge pages in use %.0f MB | peak worker memory %.1f GB (max %.1f)\n', ...
        mean(R(:,1)), mean(R(:,2)), mean(R(:,3)), mean(R(:,4))/1e6, mean(R(:,5)), mean(R(:,6)), max(R(:,6)));
    fprintf('  -> est. hours per block at 18 workers if per-call time stays this: %.2f\n', ...
        mean(R(:,1))*prod(blockSize)/18/3600);
end
delete(gcp('nocreate'));
cd(benchDir);
