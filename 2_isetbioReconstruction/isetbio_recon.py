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


def normalize_image(imageSetName, allCats, makeGrayscale, target_luminance, target_contrast, imOrig, imBorderSize):
    # Normalize the luminance and root mean square contrast of images across all categories and redefine allCats as the new normalized image directories
    new_allCats = []
    for cc in allCats:
        norm_cat = cc + '_norm'
        new_dir = f'../stimulusSets/{imageSetName}_norm/{norm_cat}/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        allImgs = os.listdir(f'../stimulusSets/{imageSetName}/{cc}/')
        allImgs = [img for img in allImgs if not img.startswith('.') and (img.endswith('.jpg') or img.endswith('.bmp') or img.endswith('.png'))]
        for ii in allImgs:
            img = cv2.imread(f'../stimulusSets/{imageSetName}/{cc}/{ii}')
            if makeGrayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (imOrig, imOrig))
            unique_elements, counts = np.unique(img, return_counts=True)
            most_frequent_index = np.argmax(counts)
            borderVal = int(unique_elements[most_frequent_index])
            replicate = cv2.copyMakeBorder(img, imBorderSize, imBorderSize, imBorderSize, imBorderSize, borderType=cv2.BORDER_CONSTANT, value=borderVal)
            replicate = replicate / 255.0
            replicate = replicate.astype(np.float32)

            # Normalize luminance and contrast
            mean_luminance = np.mean(replicate)
            std_contrast = np.std(replicate)
            normalized_img = (replicate - mean_luminance) / (std_contrast + 1e-8) * target_contrast + target_luminance
            normalized_img = np.clip(normalized_img, 0, 1)

            cv2.imwrite(f'{new_dir}{ii}', np.uint8(normalized_img*255))
        new_allCats.append(norm_cat)
    return new_allCats

def reconstruct_images(file_path, imageSetName, lbda, num_it, imOrig, imSize, imBorderSize, makeGrayscale, norm_img, target_luminance, target_contrast, species):
    # load denoiser
    main.args.model_path = './assets/conv3_ln.pt'
    model = Denoiser(main.args)
    model.load_state_dict(torch.load(main.args.model_path))
    model = model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = h5py.File(file_path, 'r')
    render_test = np.array(data['renderMtx']).astype(np.float32)
    mtx = torch.tensor(render_test.astype(np.float32).T)

    # have sparse prior file
    prior = scipy.io.loadmat('./assets/sparsePrior.mat')
    basis = np.linalg.inv(prior['regBasis'])
    sparse = SparseEstimator(device, render_test.T, basis, prior['mu'].T, lbda=lbda)

    # Initialize number of categories based on folders in konkle_imgs
    allCats = os.listdir('../stimulusSets/'+imageSetName+'/')
    allCats = [cat for cat in allCats if not cat.startswith('.')]

    if norm_img:
        allCats = normalize_image(imageSetName, allCats, makeGrayscale, target_luminance, target_contrast, imOrig, imBorderSize)
        imageSetName = imageSetName + '_norm'

    for idx1,cc in enumerate(allCats):
        print('Category: ',cc)
        allImgs = os.listdir('../stimulusSets/'+imageSetName+'/'+cc+'/')
        # exclude images that start with '.'
        allImgs = [img for img in allImgs if not img.startswith('.') and (img.endswith('.jpg') or img.endswith('.bmp') or img.endswith('.png'))]
        nImgs = len(allImgs)

        img_tor = torch.zeros((nImgs,3,int(imSize),int(imSize)))

        for idx2,ii in enumerate(allImgs):
            img = cv2.imread('../stimulusSets/'+imageSetName+'/' + cc + '/' + ii)
            # convert image to grayscale while keeping dimensions the same
            if makeGrayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img / 255.0

                img = cv2.resize(img, (imOrig, imOrig))
                # img = img.astype(np.float32)

                replicate = cv2.copyMakeBorder(img, imBorderSize, imBorderSize, imBorderSize, imBorderSize, cv2.BORDER_REPLICATE)
                img_torch = torch.tensor(replicate.astype(np.float32)).to(device)
                img_torch = img_torch.unsqueeze(0).repeat(3, 1, 1)  # Add channel dimension and repeat to match 3 channels

            else:
                img = img / 255.0

                img = cv2.resize(img, (imOrig, imOrig))
                # img = img.astype(np.float32)

                replicate = cv2.copyMakeBorder(img, imBorderSize, imBorderSize, imBorderSize, imBorderSize, cv2.BORDER_REPLICATE)
                img_torch = torch.tensor(replicate.astype(np.float32)).to(device)
                img_torch = img_torch.permute(2, 0, 1)  # Change to (C, H, W) format

            img_tor[idx2, :, :, :] = img_torch

        msmt = BayesEstimator.measure(mtx, img_tor)
        im_size = tuple([*img_tor.shape])

        # run reconstructions, can check whether objective converges & maybe change n_iter
        recon = sparse.recon(msmt.to(device), im_size, n_iter=num_it)[0]

        recon = recon.permute([0, 2, 3, 1]).cpu().numpy()

        for idx in range(recon.shape[0]):
            temp = gamma_correct(recon[idx])

            # Remove border by cropping image
            temp = temp[imBorderSize:imBorderSize+imOrig, imBorderSize:imBorderSize+imOrig, :]

            # if directory doesn't exist, create directory
            if not os.path.exists(f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName}_test/{cc}'):
                os.makedirs(f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName}_test/{cc}')
            cv2.imwrite(f'../stimulusSets/isettreeshrew/{species}_{FOV_val}/{imageSetName}_test/{cc}/{allImgs[idx]}',np.uint8(temp*255))

# initialization parameters
species = 'treeshrew'
sceneFOVdegs = 1.25
FOV_val = str(sceneFOVdegs)
imageSetName = 'Kiani_ImageSet'
sceneFOVscale = 1.2
num_it = 4000

file_path = f'/mnt/DataDrive2/treeshrew/data_raw/treeshrew_isetbio/renderMatrices/{species}/render_{FOV_val}.mat'

makeGrayscale = True
norm_img = True
target_luminance = 0.5
target_contrast = 0.1

imOrig = 227
imBorderSize = int((np.ceil(imOrig*sceneFOVscale) - imOrig)/2)
imSize = np.ceil(imOrig + imBorderSize*2)

mosaicBorderSize = ((sceneFOVscale*sceneFOVdegs) - sceneFOVdegs)/2
mosaicSize = sceneFOVdegs+mosaicBorderSize*2

if species == 'treeshrew':
    lbda = 1000
elif species == 'human':
    lbda = 1e-3

if __name__ == "__main__":
    reconstruct_images(file_path, imageSetName, lbda, num_it, imOrig, imSize, imBorderSize, makeGrayscale, norm_img, target_luminance, target_contrast, species)