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
imOrigSize = 20;
sceneFOVpadding = 1.2;
imBorder = (ceil(imOrigSize*sceneFOVpadding) - imOrigSize)/2;
imSize = ceil(imOrigSize + imBorder*2);

sceneFOVs = [2.5];

for sfd = 1:length(sceneFOVs)
    sceneFOVdegs = sceneFOVs(sfd);
    borderSize = ((sceneFOVpadding*sceneFOVdegs) - sceneFOVdegs)/2;
    mosaicSize = sceneFOVdegs+borderSize*2;

    % Only used 4x4 blocks for 1.25 & 2.5 degs and 6x6 for 5 & 10 degs
    if sceneFOVdegs<1
        nBlock = 2;
    elseif sceneFOVdegs>=1 && sceneFOVdegs<5
        nBlock = 4;
    elseif sceneFOVdegs>=5
        nBlock = 6;
    end
    blockLen = imOrigSize/nBlock;

    % Iterate through x and y block positions
    for row_idx=1:nBlock
        for col_idx=1:nBlock

            % Define indices of image that correspond to block
            if col_idx==1
                col_val = 1:ceil(col_idx*blockLen+imBorder*2);
            elseif col_idx==nBlock
                col_val = ceil((col_idx-1)*blockLen):imSize;
            else
                col_val = (col_idx-1)*blockLen:col_idx*blockLen+imBorder*2;
            end
            if row_idx==1
                row_val = 1:ceil(row_idx*blockLen+imBorder*2);
            elseif row_idx==nBlock
                row_val = ceil((row_idx-1)*blockLen):imSize;
            else
                row_val = (row_idx-1)*blockLen:row_idx*blockLen+imBorder*2;
            end

            % Calculate center of block in terms of eccentricity
            eccY = ((mean(col_val)-(imSize/2))*mosaicSize/2)/(imSize/2);
            eccX = ((mean(row_val)-(imSize/2))*mosaicSize/2)/(imSize/2);

            disp(sprintf('Image Size: %0.2f, Ecc: [%0.2f, %0.2f]',sceneFOVdegs, eccX, eccY))

            % Mosaics size initialized
            yBlockSize = (length(col_val)/imSize)*mosaicSize;
            xBlockSize = (length(row_val)/imSize)*mosaicSize;
            mosaicFOVdegs = [xBlockSize yBlockSize];

            % Integration time
            integrationTimeSeconds = 1/1000;

            % Set mean luminance
            meanLuminanceCdPerM2 = 20;

            % Set up strings for file saving
            FOV_str = num2str(sceneFOVdegs);
            eccX_str = num2str(round(eccX,2));
            eccY_str = num2str(round(eccY,2));

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
                    TSConeMosaic = cMosaicTreeShrewCreate('fovDegs', mosaicFOVdegs,'integrationTime', integrationTimeSeconds,'eccentricityDegs',[eccX,eccY]);
                    save(sprintf('%s/coneMosaics_test/treeshrew_blocked/coneMosaic_deg%s_Xecc%s_Yecc%s_quad.mat',dataSavePath,FOV_str,eccX_str,eccY_str),'TSConeMosaic')
                end

                % Enter mosaic and OI into retina object
                retina.Mosaic = TSConeMosaic;
                retina.PSF = TSoi;
            end

            % Generate test image based on block size
            blockSize = [ceil(blockLen+imBorder*2), ceil(blockLen+imBorder*2), 3];
            testImage = rand(blockSize);

            % Compute cone excitations in response to test image
            [coneExcitations, linearStimulusImage] = retina.compute(testImage);

            % Compute render matrix. May take a long time depending on
            % size of image! I have updated some parallel processing
            % allocations within ConeResponseCmosaic script.
            renderMtx = retina.forwardRender(blockSize, true, true, 'useDoublePrecision', true);
            renderMtx = double(renderMtx);

            save(sprintf('%s/renderMatrices_test/%s_blocked/render_%s_Xecc%s_Yecc%s.mat',dataSavePath,species,FOV_str,eccX_str,eccY_str),'renderMtx','-v7.3')
            delete(gcp('nocreate'))
            % end
        end
    end
end