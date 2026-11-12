%% Setup
clear all;
cd /mnt/DataDrive3/emeyer/TreeShrewObjRec/isetbio_analyses/
% Make sure startup is updated with personal path to ToolboxToolbox before
% running rest of code and check output of tbUseProject for any errors.
startup
tbUseProject('ISETTreeShrew');

%% Generate cone mosaics and render matrices piecewise
% Specify location for storing cone mosaics and render matrices. Render
% matrices can be quite large so make sure space is available.
dataSavePath = '/mnt/DataDrive2/treeshrew/data_raw/treeshrew_isetbio/';

% Specify parameters for full cone mosaic and render matrix located at a
% specific location in visual space.
species = 'treeshrew';
imOrigSize = 256;
imBorder   = 11;                 % margin px (top/left); bottom/right gets +1

% Integration time
integrationTimeSeconds = 1/1000;

% Set mean luminance
meanLuminanceCdPerM2 = 20;

% Can loop through multiple image sizes
sceneFOVs = [30];

% Parallel pool: create once and reuse for every block.
% Workers: 1 per CPU minus 2, capped at 18 (~9.5 GB each fits in 200 GB).
nCPU = str2double(getenv('SLURM_CPUS_PER_TASK'));
if isnan(nCPU)
    nWorkers = 3;                      % not under Slurm: interactive default
else
    nWorkers = min(18, nCPU - 2);
end

pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= nWorkers
    delete(pool);
    c = parcluster('Processes');
    js = getenv('MATLAB_JOBDIR');
    if ~isempty(js), c.JobStorageLocation = js; end
    c.NumWorkers = nWorkers;
    parpool(c, nWorkers);
end
fprintf('Using %d workers\n', nWorkers);


for sfd = 1:length(sceneFOVs)
    sceneFOVdegs = sceneFOVs(sfd);
    nBlock     = 8;
    blockLen   = imOrigSize/nBlock;                 % 32
    blockSize1 = blockLen + 2*imBorder + 1;         % 55, odd
    imSize     = imOrigSize + 2*imBorder + 1;       % 279
    pixDeg     = sceneFOVdegs / imOrigSize;
    imCenter   = imBorder + (imOrigSize+1)/2;       % 139.5
    % Iterate through x and y block positions
    for row_idx=1:nBlock
        for col_idx=1:nBlock

            % Define indices of image that correspond to block
            col_val = (col_idx-1)*blockLen + (1:blockSize1);
            row_val = (row_idx-1)*blockLen + (1:blockSize1);
            eccX =  (mean(col_val) - imCenter) * pixDeg;
            eccY = -(mean(row_val) - imCenter) * pixDeg;
            xBlockSize = length(col_val)*pixDeg;
            yBlockSize = length(row_val)*pixDeg;
            blockSize  = [blockSize1, blockSize1, 3];

            disp(sprintf('Image Size: %0.2f, Ecc: [%0.2f, %0.2f]',sceneFOVdegs, eccX, eccY))

            mosaicFOVdegs = [xBlockSize yBlockSize];

            % Set up strings for file saving
            FOV_str = num2str(sceneFOVdegs);
            eccX_str = num2str(round(eccX,2));
            eccY_str = num2str(round(eccY,2));

            renderFile = sprintf('%s/renderMatrices_test/%s_blocked/render_%s_Xecc%s_Yecc%s.mat', ...
                dataSavePath,species,FOV_str,eccX_str,eccY_str);
            if exist(renderFile,'file'), fprintf('  exists, skipping\n'); continue; end

            % Initialize retina object
            retina = ConeResponseCmosaic(eccX,eccY,'fovealDegree',mosaicFOVdegs(1));

            if strcmp(species,'human')
                % Generate human optical image
                HumanOI = oiCreate('wvf human');

                % Load or create cone mosaic
                try
                    load(sprintf('%s/coneMosaics_test/human_blocked/coneMosaic_deg%s_Xecc%s_Yecc%s_quad.mat',dataSavePath,FOV_str,eccX_str,eccY_str),'HumanConeMosaic')
                catch
                    HumanConeMosaic = cMosaic('sizeDegs',mosaicFOVdegs,'eccentricityDegs',[eccX,eccY]);
                    save(sprintf('%s/coneMosaics_test/human_blocked/coneMosaic_deg%s_Xecc%s_Yecc%s_quad.mat',dataSavePath,FOV_str,eccX_str,eccY_str),'HumanConeMosaic')
                end

                % Enter mosaic and OI into retina object. Could also
                % specify display params as retina.Display but defaults are
                % used here.
                retina.Mosaic = HumanConeMosaic;
                retina.PSF = HumanOI;

            elseif strcmp(species,'treeshrew')
                % Generate tree shrew optical image
                TSoi = oiTreeShrewCreate('opticsType','wvf','name','wvf-based optics');

                % Load or create cone mosaic
                try
                    load(sprintf('%s/coneMosaics_test/treeshrew_blocked/coneMosaic_deg%s_Xecc%s_Yecc%s_quad.mat',dataSavePath,FOV_str,eccX_str,eccY_str),'TSConeMosaic')
                catch
                    TSConeMosaic = cMosaicTreeShrewCreate('sizeDegs', mosaicFOVdegs,'integrationTime', integrationTimeSeconds,'eccentricityDegs',[eccX,eccY]);
                    save(sprintf('%s/coneMosaics_test/treeshrew_blocked/coneMosaic_deg%s_Xecc%s_Yecc%s_quad.mat',dataSavePath,FOV_str,eccX_str,eccY_str),'TSConeMosaic')
                end

                % Enter mosaic and OI into retina object
                retina.Mosaic = TSConeMosaic;
                retina.PSF = TSoi;
                retina.psfData = [];
            end

            % Compute render matrix. May take a long time depending on
            % size of image! See create_render_matrix for details
            % regarding parallel processing constraints
            renderMtx = retina.forwardRender(blockSize, false, false, ...
                'useDoublePrecision', true, 'numWorkers', nWorkers);

            save(renderFile,'renderMtx','-v7.3')
            fprintf('%s  saved %s\n', datestr(now), renderFile);

            % end
        end
    end
end