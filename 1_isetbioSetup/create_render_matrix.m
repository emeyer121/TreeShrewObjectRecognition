%% Setup and pull from ISETTreeShrew
clear all;

% Make sure startup is updated with personal path to ToolboxToolbox before
% running rest of code and check output of tbUseProject for any errors.
startup
tbUseProject('ISETTreeShrew');

% Specify location for storing cone mosaics and render matrices. Render
% matrices can be quite large so make sure space is available.
dataSavePath = '/mnt/DataDrive2/treeshrew/data_raw/treeshrew_isetbio/';

%% Specify parameters 
% Set parameters for full cone mosaic and render matrix located at a
% specific location in visual space and create retina object

% Set species, either 'treeshrew' or 'human'
species = 'treeshrew';

% Image dimensions in pixels that you want to reconstruct
imOrigSize = 20;

% Set mosaic size in visual degrees
sceneFOVdegs = 0.25;
sceneFOVpadding = 1.2;

% compute mosaic size and image size with padding
imageSize = [round(imOrigSize*sceneFOVpadding), round(imOrigSize*sceneFOVpadding), 3];
mosaicFOVdegs = [sceneFOVdegs*sceneFOVpadding sceneFOVdegs*sceneFOVpadding];

% Set eccentricity of mosaic
eccX = 0;
eccY = 0;

% Integration time
integrationTimeSeconds = 1/1000;

% Set mean luminance
meanLuminanceCdPerM2 = 20;

% Build retina object based on size and location
retina = ConeResponseCmosaic(eccX,eccY,'fovealDegree',sceneFOVdegs*sceneFOVpadding);

% Set string for save paths
FOV_str = num2str(sceneFOVdegs);

if strcmp(species,'human')
    % Generate human optical image
    HumanOI = oiCreate('wvf human');

    % Load or create cone mosaic
    try
        load([dataSavePath,'coneMosaics_test/human/coneMosaic_deg',FOV_str,'.mat'],'HumanConeMosaic')
    catch
        HumanConeMosaic = cMosaic('sizeDegs',mosaicFOVdegs,'eccentricityDegs',[eccX,eccY]);
        save([dataSavePath,'/coneMosaics_test/human/coneMosaic_deg',FOV_str,'.mat'],'HumanConeMosaic')
    end

    % Enter mosaic and OI into retina object. Could also specify display
    % params as retina.Display but defaults are used here.
    retina.Mosaic = HumanConeMosaic;
    retina.PSF = HumanOI;

elseif strcmp(species,'treeshrew')
    % Generate tree shrew optical image
    TSOI = oiTreeShrewCreate('opticsType','wvf','name','wvf-based optics');

    % Load or create cone mosaic
    try
        load([dataSavePath,'/coneMosaics_test/treeshrew/coneMosaic_deg',FOV_str,'.mat'],'TSConeMosaic')
    catch
        TSConeMosaic = cMosaicTreeShrewCreate('fovDegs', mosaicFOVdegs,'integrationTime', integrationTimeSeconds,'eccentricityDegs',[eccX,eccY]);
        save([dataSavePath,'/coneMosaics_test/treeshrew/coneMosaic_deg',FOV_str,'.mat'],'TSConeMosaic')
    end

    % Enter mosaic and OI into retina object. Could also specify display
    % params as retina.Display but defaults are used here.
    retina.Mosaic = TSConeMosaic;
    retina.PSF = TSOI;
end

%% Compute render matrix. 
% May take a long time depending on size of image!
% If you want to update parallel processing allocations, go into 
% ConeResponseCmosaic script and add the following lines after parameter
% parsing. Can easily overload a system and crash otherwise:
% maxNumCompThreads(N);  % Limit each MATLAB worker to N threads
% parpool(N);           % Now start your parallel pool with N workers
% Also update in forwardRender: parfor (idx = 1:length(testLinear(:)), N)
renderMtx = retina.forwardRender(imageSize, true, true, 'useDoublePrecision', true);
renderMtx = double(renderMtx);
save([dataSavePath,'renderMatrices_test/',species,'/render_',FOV_str,'.mat'],'renderMtx','-v7.3')