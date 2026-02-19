# setup and load libraries
import cupy as cp, numpy as np
import main, torch, scipy.io, h5py
import matplotlib.pyplot as plt
from inverse.bayes import *
from inverse.solver import RenderMatrix, linear_inverse
from utils.dataset import DataSet, gamma_correct
from utils.sampling import forward_matrix
from models.denoiser import Denoiser
from tqdm.notebook import tqdm
import os, cv2
from collections import Counter
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
# Thread-based parallelization (if GPU memory is a concern)
from concurrent.futures import ThreadPoolExecutor

def normalize_image(imageSetName, allCats, makeGrayscale, target_luminance, target_contrast, imOrig, imBorderSize):
    # Normalize the luminance and root mean square contrast of images across all categories and redefine allCats as the new normalized image directories
    new_allCats = []
    # Iterate through image categories
    for cc in allCats:
        norm_cat = cc + '_norm'
        new_dir = f'../stimulusSets/{imageSetName}_norm/{norm_cat}/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        # Iterate through images in the category
        allImgs = os.listdir(f'../stimulusSets/{imageSetName}/{cc}/')
        allImgs = [img for img in allImgs if not img.startswith('.') and (img.endswith('.jpg') or img.endswith('.bmp') or img.endswith('.png'))]
        for ii in allImgs:
            img = cv2.imread(f'../stimulusSets/{imageSetName}/{cc}/{ii}')
            # Can specity whether to convert to grayscale or keep as RGB
            if makeGrayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (imOrig, imOrig))

            # Normalize luminance and contrast
            mean_luminance = np.mean(img)
            std_contrast = np.std(img)
            normalized_img = (img - mean_luminance) / (std_contrast + 1e-8) * target_contrast + target_luminance
            normalized_img = np.clip(normalized_img, 0, 1)

            # Save the normalized image in the new directory
            cv2.imwrite(f'{new_dir}{ii}', np.uint8(normalized_img*255))
        # Specify new category name for normalized images and output list
        new_allCats.append(norm_cat)
    return new_allCats

def process_block(args):
    """Process a single block - this function will be called in parallel"""
    row_idx, col_idx, species, sceneFOVdegs, imageSetName, mosaicSize, imSize, blockLen, imBorder, blockSize, imOrig, makeGrayscale, target_luminance, target_contrast, norm_img, renderLoadPath = args
    
    # Calculate column indices corresponding to block
    if col_idx == 1:
        col_val = np.arange(1,np.ceil(col_idx*blockLen+imBorder*2)+1)
    elif col_idx==nBlock:
        col_val = np.arange(np.ceil((col_idx-1)*blockLen), imSize+1)
    else:
        col_val = np.arange((col_idx-1)*blockLen,col_idx*blockLen+imBorder*2)

    # Calculate row indices corresponding to block
    if row_idx == 1:
        row_val = np.arange(1,np.ceil(row_idx*blockLen+imBorder*2)+1)
    elif row_idx==nBlock:
        row_val = np.arange(np.ceil((row_idx-1)*blockLen), imSize+1)
    else:
        row_val = np.arange((row_idx-1)*blockLen,row_idx*blockLen+imBorder*2)

    # Compute eccentricity values for specific block center
    eccY = ((np.mean(col_val)-(imSize/2))*mosaicSize/2)/(imSize/2)
    eccX = ((np.mean(row_val)-(imSize/2))*mosaicSize/2)/(imSize/2)
    FOV_val = str(sceneFOVdegs)

    eccX_val = str(np.round(eccX,2))
    eccY_val = str(np.round(eccY,2))

    # Recommendation for lambda values based on species and scene FOV
    if species == 'human':
        if sceneFOVdegs <= 2:
            lbda = 1e-3
        elif sceneFOVdegs <= 5 and sceneFOVdegs > 2:
            lbda = 1e-2
        elif sceneFOVdegs > 5:
            lbda = 1e-1
    elif species == 'treeshrew':
        if sceneFOVdegs < 5:
            lbda = 1e-2
        elif sceneFOVdegs >= 5:
            lbda = 1e-2

    file_path = f'{renderLoadPath}{species}_blocked/render_{FOV_val}_Xecc{eccX_val}_Yecc{eccY_val}.mat'

    # load denoiser
    main.args.model_path = './assets/conv3_ln.pt'
    model = Denoiser(main.args)
    model.load_state_dict(torch.load(main.args.model_path))
    model = model.eval()

    # Set device for computation (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing block ({row_idx}, {col_idx}) on device: {device}")

    # Load render matrix (created in 1_isetbioSetup folder) and convert to torch tensor
    data = h5py.File(file_path, 'r')
    render_test = np.array(data['renderMtx']).astype(np.float32)
    mtx = torch.tensor(render_test.astype(np.float32).T)

    # Load sparse prior and initialize sparse estimator
    prior = scipy.io.loadmat('./assets/sparsePrior.mat')
    basis = np.linalg.inv(prior['regBasis'])
    sparse = SparseEstimator(device, render_test.T, basis, prior['mu'].T, lbda=lbda)

    # Initialize number of categories based on folders in imageSetName directory
    allCats = os.listdir('../stimulusSets/'+imageSetName+'/')
    allCats = [cat for cat in allCats if not cat.startswith('.')]

    # Option to normalize image luminance and contrast across categories
    if norm_img:
        allCats = normalize_image(imageSetName, allCats, makeGrayscale, target_luminance, target_contrast, imOrig, imBorder)
        imageSetName = imageSetName + '_norm'

    # Iterate through categories and load images
    for idx1,cc in enumerate(allCats):
        print(f'Block ({row_idx}, {col_idx}) - Category: {cc}')
        allImgs = os.listdir('../stimulusSets/'+imageSetName+'/'+cc+'/')
        # exclude images that start with '.'
        allImgs = [img for img in allImgs if not img.startswith('.') and (img.endswith('.jpg') or img.endswith('.bmp') or img.endswith('.png'))]
        allImgs = allImgs[201:301]
        nImgs = len(allImgs)

        # Set up tensor that will be used for reconstruction of all images within category
        img_tor = torch.zeros((nImgs,blockSize[2],blockSize[0],blockSize[1]))

        # Iterate through all images in category, load and preprocess
        for idx2,ii in enumerate(allImgs):
            img = cv2.imread('../stimulusSets/'+imageSetName+'/' + cc + '/' + ii)
            # convert image to grayscale and add border by replicating edges (maybe change this in the future)
            # repeat channel dimensions if grayscale to keep dims the same
            if makeGrayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (imOrig, imOrig))

                unique_elements, counts = np.unique(img, return_counts=True)
                most_frequent_index = np.argmax(counts)
                borderVal = int(unique_elements[most_frequent_index])
                replicate = cv2.copyMakeBorder(img, imBorder, imBorder, imBorder, imBorder, borderType=cv2.BORDER_CONSTANT, value=borderVal)
                replicate = replicate / 255.0

                replicate = replicate.astype(np.float32)
                crop_img = replicate[row_val.astype(int)-1, :][:, col_val.astype(int)-1]

                if np.shape(crop_img)[0] - blockSize[0] == 1:
                    crop_img = crop_img[:int(blockSize[0]), :]
                if np.shape(crop_img)[1] - blockSize[1] == 1:
                    crop_img = crop_img[:, :int(blockSize[1])]

                img_torch = torch.tensor(crop_img.astype(np.float32)).to(device)
                img_torch = img_torch.unsqueeze(0).repeat(3, 1, 1)  # Add channel dimension and repeat to match 3 channels
            else:

                # If keeping aas RGB, just add border by replicating edges and convert to torch tensor
                # Need to test this******
                img = cv2.resize(img, (imOrig, imOrig))

                unique_elements, counts = np.unique(img, return_counts=True)
                most_frequent_index = np.argmax(counts)
                borderVal = int(unique_elements[most_frequent_index])
                replicate = cv2.copyMakeBorder(img, imBorder, imBorder, imBorder, imBorder, borderType=cv2.BORDER_CONSTANT, value=borderVal)
                replicate = replicate / 255.0

                replicate = replicate.astype(np.float32)
                crop_img = replicate[row_val.astype(int)-1, :][:, col_val.astype(int)-1]

                if np.shape(crop_img)[0] - blockSize[0] == 1:
                    crop_img = crop_img[:int(blockSize[0]), :]
                if np.shape(crop_img)[1] - blockSize[1] == 1:
                    crop_img = crop_img[:, :int(blockSize[1])]

                img_torch = torch.tensor(replicate.astype(np.float32)).to(device)
                img_torch = img_torch.permute(2, 0, 1)  # Change to (C, H, W) format

            # Add preprocessed image to tensor for reconstruction
            img_tor[idx2, :, :, :] = img_torch

        msmt = BayesEstimator.measure(mtx, img_tor)
        im_size = tuple([*img_tor.shape])

        # run reconstructions, can check whether objective converges & maybe change n_iter and/or lambda
        recon = sparse.recon(msmt.to(device), im_size, n_iter=num_it, print_loss=False)[0]
        recon = recon.permute([0, 2, 3, 1]).cpu().numpy()

        # Iterate through reconstructions, apply gamma correction, then save images in new directory organized by category/block/images
        for idx in range(recon.shape[0]):
            temp = gamma_correct(recon[idx])

            if not os.path.exists(f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName}_quad/{cc}/Xecc{eccX_val}_Yecc{eccY_val}'):
                os.makedirs(f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName}_quad/{cc}/Xecc{eccX_val}_Yecc{eccY_val}')
            cv2.imwrite(f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName}_quad/{cc}/Xecc{eccX_val}_Yecc{eccY_val}/{allImgs[idx]}',np.uint8(temp*255))
    
    return f"Block ({row_idx}, {col_idx}) completed"

def run_parallel_threads():
    # Create list of arguments for each block
    block_args = []
    for row_idx in range(1, nBlock+1):
        for col_idx in range(1, nBlock+1):
            args = (row_idx, col_idx, species, sceneFOVdegs, imageSetName, mosaicSize, imSize, blockLen, imBorder, 
                    blockSize, imOrig, makeGrayscale, target_luminance, target_contrast, norm_img, renderLoadPath)
            block_args.append(args)
    
    # Use ThreadPoolExecutor (better for GPU-bound tasks)
    n_threads = 6 # Limit threads to avoid GPU memory issues
    print(f"Using {n_threads} threads for {len(block_args)} blocks")
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_block = {executor.submit(process_block, args): args for args in block_args}
        
        for future in as_completed(future_to_block):
            args = future_to_block[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'Block {args[0:2]} generated an exception: {exc}')

def merge_blocks():

    # Iterate through categories (either normalized or original)
    allCats = os.listdir('../stimulusSets/'+imageSetName+'/')
    allCats = [cat for cat in allCats if not cat.startswith('.')]

    for idx1,cc in enumerate(allCats):

        new_dir = f'../stimulusSets/{imageSetName}/{cc}'
        norm_cat = cc
        if norm_img:
            imageSetName_new = imageSetName + '_norm'
            norm_cat = cc + '_norm'
            new_dir = f'../stimulusSets/{imageSetName_new}/{norm_cat}/'
        else:
            imageSetName_new = imageSetName

        # Find all images in category and iterate through them
        allImgs = os.listdir(new_dir)
        # exclude images that start with '.'
        allImgs = [img for img in allImgs if not img.startswith('.') and (img.endswith('.jpg') or img.endswith('.bmp') or img.endswith('.png'))]
        
        for idx,ii in enumerate(allImgs):
            init_img = np.zeros((imOrig, imOrig, 3), dtype=np.float32)

            # Iterate through blocks and merge reconstructions by cropping borders and stitching together
            for row_idx in range(1, nBlock+1):
                for col_idx in range(1, nBlock+1):

                    # Calculate column indices corresponding to block
                    if col_idx == 1:
                        col_val = np.arange(1,np.ceil(col_idx*blockLen+imBorder*2)+1)
                    elif col_idx==nBlock:
                        col_val = np.arange(np.ceil((col_idx-1)*blockLen), imSize+1)
                    else:
                        col_val = np.arange((col_idx-1)*blockLen,col_idx*blockLen+imBorder*2)

                    # Calculate row indices corresponding to block
                    if row_idx == 1:
                        row_val = np.arange(1,np.ceil(row_idx*blockLen+imBorder*2)+1)
                    elif row_idx==nBlock:
                        row_val = np.arange(np.ceil((row_idx-1)*blockLen), imSize+1)
                    else:
                        row_val = np.arange((row_idx-1)*blockLen,row_idx*blockLen+imBorder*2)

                    col_block = np.arange((col_idx-1)*blockLen, col_idx*blockLen).astype(int)
                    row_block = np.arange((row_idx-1)*blockLen, row_idx*blockLen).astype(int)

                    # Compute eccentricity values for specific block center
                    eccY = ((np.mean(col_val)-(imSize/2))*mosaicSize/2)/(imSize/2)
                    eccX = ((np.mean(row_val)-(imSize/2))*mosaicSize/2)/(imSize/2)
                    FOV_val = str(sceneFOVdegs)

                    eccX_val = str(np.round(eccX,2))
                    eccY_val = str(np.round(eccY,2))

                    # Load reconstructed block image
                    file_path = f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName_new}_quad/{norm_cat}/Xecc{eccX_val}_Yecc{eccY_val}/{allImgs[idx]}'

                    img = cv2.imread(file_path)
                    img_size = img.shape[0]

                    img_crop = img[imBorder:img_size-imBorder, imBorder:img_size-imBorder, :]

                    # Place cropped block reconstruction in correct location within initialized image
                    init_img[np.ix_(row_block, col_block)] = img_crop

            # Save merged image in new directory organized by category/merged images
            if not os.path.exists(f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName_new}_quad/{norm_cat}/merged'):
                os.makedirs(f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName_new}_quad/{norm_cat}/merged')
            cv2.imwrite(f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName_new}_quad/{norm_cat}/merged/{allImgs[idx]}',np.uint8(init_img))

# initialize species, image sizes
torch.cuda.set_device(1)
species = 'treeshrew'
sceneFOVdegs = 10
imageSetName = 'Camel_v2_test_nn/original'
renderLoadPath = '/mnt/DataDrive2/treeshrew/data_raw/treeshrew_isetbio/renderMatrices/'
sceneFOVscale = 1.2
imOrig = 227
imBorder = int((np.ceil(imOrig*sceneFOVscale) - imOrig)/2)
imSize = np.ceil(imOrig + imBorder*2)

borderSize = ((sceneFOVscale*sceneFOVdegs) - sceneFOVdegs)/2
mosaicSize = sceneFOVdegs+borderSize*2

# Initialize reconstruction iterations
num_it = 4000

# Parameters for preprocessing images
makeGrayscale = True
norm_img = False
target_luminance = 0.5
target_contrast = 0.1

# Set number of blocks based on mosaic size
if sceneFOVdegs < 1:
    nBlock = 2
elif sceneFOVdegs >= 1 and sceneFOVdegs < 5:
    nBlock = 4
elif sceneFOVdegs >=5:
    nBlock = 6

blockLen = imOrig/nBlock
blockSize = [int(np.ceil(blockLen+imBorder*2)), int(np.ceil(blockLen+imBorder*2)), 3]

# Run the parallelized version
if __name__ == '__main__':
    # ThreadPoolExecutor (better for GPU-bound tasks)
    run_parallel_threads()
    merge_blocks()