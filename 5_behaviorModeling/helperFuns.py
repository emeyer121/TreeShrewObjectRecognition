import pandas as  pd
import scipy.io as sio
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import cdist
import torch

def get_train_targdist_data(datafilepath,task):
    df = pd.read_csv(datafilepath + task + ".csv")
    temp = sio.loadmat(datafilepath+"tablecolumns.mat", struct_as_record=False, squeeze_me=True)
    column_names = temp['columns']

    # Convert to a NumPy array
    numpy_array = df.to_numpy()

    if task == 'Camel_Rhino_test_nn':
         shrewID = 3
        # shrewID = np.unique(numpy_array[:,column_names=='ShrewID'])
    else:
         shrewID = np.unique(numpy_array[:,column_names=='ShrewID'])

    allShrewID = numpy_array[:,column_names=='ShrewID']
    shrewID_idx = np.any(allShrewID==shrewID,axis=1)

    targ_idx = np.unique(numpy_array[shrewID_idx,column_names=='T_Expt_ID'])
    test_targ_idx = np.unique(numpy_array[shrewID_idx,column_names=='TestTarg'])
    nov_targ_idx = np.unique(numpy_array[shrewID_idx,column_names=='NovelTarg'])

    # Get targ_idx numbers that exclude those in test_targ_idx and nov_targ_id
    train_targ_idx = np.setdiff1d(targ_idx, np.union1d(test_targ_idx, nov_targ_idx))
    # Remove target 9 from camel_v2_test_nn since it is only used in a few sessions
    if task == 'Camel_v2_test_nn':
        train_targ_idx = train_targ_idx[train_targ_idx != 9]

    print(train_targ_idx)

    dist_idx = np.unique(numpy_array[shrewID_idx,column_names=='D_Expt_ID'])
    test_dist_idx = np.unique(numpy_array[shrewID_idx,column_names=='TestDist'])
    nov_dist_idx = np.unique(numpy_array[shrewID_idx,column_names=='NovelDist'])

    # Get targ_idx numbers that exclude those in test_targ_idx and nov_targ_id
    train_dist_idx = np.setdiff1d(dist_idx, np.union1d(test_dist_idx, nov_dist_idx))
    print(train_dist_idx)

    avg_perf = np.full((len(train_targ_idx), len(train_dist_idx)), np.nan)
    for tt_idx, tt in enumerate(train_targ_idx):
        for dd_idx, dd in enumerate(train_dist_idx):
                if task == 'Camel_Rhino_test_nn':
                    avg_perf[tt_idx, dd_idx] = np.nanmean(
                        numpy_array[:,column_names=='correct'][
                            (numpy_array[:,column_names=='T_Expt_ID'] == tt) &
                            (numpy_array[:,column_names=='D_Expt_ID'] == dd) &
                            (numpy_array[:,column_names=='ShrewID'] == 3)])
                else:
                     avg_perf[tt_idx, dd_idx] = np.nanmean(
                        numpy_array[:,column_names=='correct'][
                            (numpy_array[:,column_names=='T_Expt_ID'] == tt) &
                            (numpy_array[:,column_names=='D_Expt_ID'] == dd)])

    return train_targ_idx, train_dist_idx, avg_perf

def structure_isetbio_data(img_files, Y_labels, unique_labels, input_size):
    """
    Function to load and preprocess images to fit the input size of the deep neural network (DNN).
    
    Parameters:
    img_folder : list
        List of image paths.
    
    Returns:
    DNN_input : numpy array
        Preprocessed images ready for input into the neural network.
    """

    n_imgs = len(img_files)
    
    # Initialize an array to store all preprocessed images
    DNN_input = np.zeros((input_size[0], input_size[1], 3, n_imgs))

    for ii in range(n_imgs):
        # Load the image using OpenCV
        # img_path = os.path.join(img_folder[ii])
        # temp = cv2.imread(img_folder[ii])
        temp = Image.open(img_files[ii]) 
        
        if temp is None:
            raise FileNotFoundError(f"Image {img_files[ii]} not found.")
        
        center_crop = transforms.CenterCrop(input_size)

        temp_resized = center_crop(temp)
        temp_resized_array = np.array(temp_resized)
        
        # img_size = temp.shape

        if len(temp_resized_array.shape) == 3:  # If the image has 3 channels (RGB)
            if temp_resized_array.shape[2] > 3: 
                temp_resized_array = temp_resized_array[:,:, :3]  # Keep only the first 3 channels if there are more than 3
            DNN_input[:, :, :, ii] = temp_resized_array
        else:  # If it's a grayscale image
            # Resize and replicate the grayscale image across 3 channels
            temp_resized_array = np.stack([temp_resized_array] * 3, axis=-1)
            DNN_input[:, :, :, ii] = temp_resized_array

        # Split DNN_input into two variables based on whether image name corresponds to unique_labels[0] or unique_labels[1]
        DNN_input_targ = DNN_input[:, :, :, [i for i, label in enumerate(Y_labels) if label == unique_labels[0]]]
        DNN_input_dist = DNN_input[:, :, :, [i for i, label in enumerate(Y_labels) if label == unique_labels[1]]]
        

    return DNN_input_targ, DNN_input_dist

# Define a hook function to capture the activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu().numpy()
    return hook

def register_hooks(model, layers):
    for layer in layers:
        layer_module = dict(model.named_modules())[layer]
        layer_module.register_forward_hook(get_activation(layer))

def get_activations(model, images, layers, device):
    # Clear previous activations
    global activations
    activations = {layer: [] for layer in layers}
    
    # Register hooks on the specified layers
    register_hooks(model, layers)
    
    all_activations = {layer: [] for layer in layers}
    for i in range(images.shape[3]):
        image_slice = images[:, :, :, i]
        image_slice = torch.from_numpy(image_slice).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)  # Convert to tensor and permute to (N, C, H, W)
        
        model(image_slice)
        for layer in layers:
            all_activations[layer].append(activations[layer])
    for layer in layers:
        all_activations[layer] = np.array(all_activations[layer])
    return all_activations


def compute_distances(taskName, activations_dict, model_layers):
    distances_dict = {}
    
    for layer in model_layers:
        # Extract activations for the current layer
        activations = activations_dict[layer]
        # print(f"Layer {layer} activations shape: {activations.shape}")

        # Reshape the activations to 2D (flatten the spatial dimensions)
        if len(activations.shape) == 5:
            n_samples, _, n_channels, height, width = activations.shape
            activations = activations.reshape(n_samples, n_channels * height * width)
        else:
            activations = np.squeeze(activations)
            
        # Separate target and distractor activations
        if taskName =='Camel_v2_test_nn' or taskName == 'Camel_Rhino_test_nn':
            target_activations = activations[:11]
            # print(target_activations.shape)
            distractor_activations = activations[11:19]
            # print(distractor_activations.shape)
        elif taskName == 'Camel_background_matrix':
            target_activations = activations[:18]
            # print(target_activations.shape)
            distractor_activations = activations[18:36]
            # print(distractor_activations.shape)


        # Compute distances between all target and distractor activations
        distances = cdist(target_activations, 
                          distractor_activations, 
                          metric='euclidean')
        
        distances_dict[layer] = distances
    
    return distances_dict

def compute_correlation(avg_perf, distances_dict):
    correlation_dict = {}
    correlation = {}
    ci_lower = {}
    ci_upper = {}

    shuff_lower = {}
    shuff_upper = {}
    
    for layer, distances in distances_dict.items():
        # Flatten the distances and avg_perf arrays
        distances_flat = distances.flatten()
        avg_perf_flat = avg_perf.flatten()
        
        # Compute the correlation coefficient
        print(distances.shape)
        print(avg_perf.shape)
        avg_perf_flat_nan = avg_perf_flat[~np.isnan(avg_perf_flat)]
        distances_flat_nan = distances_flat[~np.isnan(avg_perf_flat)]
        correlation[layer] = np.corrcoef(distances_flat_nan, avg_perf_flat_nan)[0, 1]

        # Perform resampling to compute error bars for the correlation
        n_samples = len(avg_perf_flat_nan)
        n_resamples = 1000
        resampled_correlations = np.zeros(n_resamples)
        for i in range(n_resamples):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            resampled_avg_perf = avg_perf_flat_nan[indices]
            resampled_distances = distances_flat_nan[indices]
            resampled_correlations[i] = np.corrcoef(resampled_avg_perf, resampled_distances)[0, 1]

        # Compute null distribution by shuffling label indices
        shuffled_correlations = np.zeros(n_resamples)
        for i in range(n_resamples):
            shuffled_indices = np.random.permutation(n_samples)
            shuffled_distances = distances_flat_nan[shuffled_indices]
            shuffled_correlations[i] = np.corrcoef(avg_perf_flat_nan, shuffled_distances)[0, 1]

        # Compute the mean and standard deviation of the resampled correlations
        mean_correlation = np.mean(resampled_correlations)
        std_correlation = np.std(resampled_correlations)
        # Compute the 95% confidence interval
        ci_lower[layer] = np.percentile(resampled_correlations, 2.5)
        ci_upper[layer] = np.percentile(resampled_correlations, 97.5)

        # Compute the 95% confidence interval
        shuff_lower[layer] = np.percentile(shuffled_correlations, 2.5)
        shuff_upper[layer] = np.percentile(shuffled_correlations, 97.5)
        
        # Store the correlation in the dictionary
        correlation_dict[layer] = mean_correlation
    
    return correlation, ci_lower, ci_upper, shuff_lower, shuff_upper