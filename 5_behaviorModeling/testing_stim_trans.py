"""
Stimulus Transformations Library

A library for stimulus image transformations including centering, 
scaling, texture synthesis, skeletonization, and neural network feature extraction.

This module provides functions for:
- Object centering and alignment
- Object scaling to match reference dimensions  
- Texture synthesis using Portilla-Simoncelli model
- Object skeletonization
- Neural network feature extraction

"""

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import cv2
from scipy import stats, ndimage
import matplotlib.pyplot as plt
import plenoptic as po
import torch
import os
from skimage.morphology import skeletonize, remove_small_holes
from torchvision import models
from mouse_vision.core.model_loader_utils import load_model
from mouse_vision.models.model_paths import MODEL_PATHS
from skimage.measure import label, regionprops

def transform_image(img_test: np.ndarray, operation: str, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]:
    """
    Main entry point for image operations.
    
    This function provides a unified interface for many image transformation operations
    including centering, scaling, texture synthesis, skeletonization, and neural
    network feature extraction.
    
    Parameters
    ----------
    img_test : np.ndarray
        Input image to operate on. Should be a 2D numpy array (grayscale).
    operation : str
        Type of operation to perform. Must be one of:
        - 'center': Center image to reference or image center
        - 'scale': Scale image to match reference dimensions
        - 'rotate': Rotate image plane to match reference long axis orientation
        - 'texture_inplace': Generate texture metamer in place
        - 'texture_crop': Generate texture metamer with cropping
        - 'skeleton': Extract object skeleton
        - 'NN': Extract neural network features
    **kwargs : dict
        Operation-specific parameters (see below).
        
    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]
        The result depends on the operation:
        - 'center', 'scale', 'rotate', 'texture_inplace': Single numpy array
        - 'texture_crop': Tuple of (cropped_image, texture_image)
        - 'skeleton': Tuple of (binary_mask, skeleton_image)
        - 'NN': List of feature arrays
        
    Operation-specific Parameters
    -----------------------------
    center:
        img_ref : np.ndarray, optional
            Reference image for alignment. If not provided, centers to image center.
            
    scale:
        img_ref : np.ndarray, required
            Reference image that test image is scaled to match.

    rotate:
        img_ref : np.ndarray, required
            Reference image that test image is rotated to match.
            
    texture_inplace:
        n_scales : int, default=2
            Number of scales for Portilla-Simoncelli model.
        max_iter : int, default=1500
            Maximum iterations for synthesis.
        device : str, default='auto'
            Device to use: 'cuda', 'cpu', or 'auto'.
            
    texture_crop:
        n_scales : int, default=2
            Number of scales for Portilla-Simoncelli model.
        target_size : int, default=256
            Size to resize cropped image to.
        max_iter : int, default=1500
            Maximum iterations for synthesis.
        device : str, default='auto'
            Device to use: 'cuda', 'cpu', or 'auto'.
            
    skeleton:
        area_threshold : int, default=3
            Area threshold for removing small holes.
        blur_kernel : Tuple[int, int], default=(5, 5)
            Kernel size for Gaussian blur.
        invert : bool, default=True
            Whether to invert colors at the end.
            
    NN:
        network : str, default='alexnet'
            Network architecture. Supported: 'alexnet', 'vgg11', 'vgg16', 
            'resnet18', 'resnet50'.
        layer_types : List[str], default=['Conv2d', 'Linear']
            Layer types to extract features from.
        device : str, default='auto'
            Device to use: 'cuda', 'cpu', or 'auto'.
            
    Raises
    ------
    ValueError
        If operation is not recognized or required parameters are missing.
        
    """

    if operation == 'center':
        return center_image(img_test, **kwargs)
    
    elif operation == 'scale':
        img_ref = kwargs.get('img_ref')
        if img_ref is None:
            raise ValueError("img_ref is required for 'scale' operation")
        return scale_image(img_test, img_ref)
    
    elif operation == 'rotate':
        img_ref = kwargs.get('img_ref')
        if img_ref is None:
            raise ValueError("img_ref is required for 'scale' operation")
        return rotate_image(img_test, img_ref)
    
    elif operation == 'texture_inplace':
        return texture_inplace(img_test, **kwargs)
    
    elif operation == 'texture_crop':
        return texture_crop(img_test, **kwargs)
    
    elif operation == 'skeleton':
        return skeletonize_object(img_test, **kwargs)
    
    elif operation == 'NN':
        return NN_activation(img_test, **kwargs)
    
    else:
        raise ValueError(f"Unknown operation: {operation}")

def center_image(img_test: np.ndarray, img_ref: Optional[np.ndarray] = None) -> np.ndarray:
    
    backgroundTest = stats.mode(img_test.flatten())[0]
    binary = img_test != backgroundTest
    binary = binary.astype(np.uint8) * 255  # Convert boolean to uint8 for display

    # Calculate moments
    M = cv2.moments(binary)

    if M["m00"] != 0:
        cX_old = int(M["m10"] / M["m00"])
        cY_old = int(M["m01"] / M["m00"])
    else:
        raise ValueError("The test image is empty or has no foreground pixels.")
    
    # Threshold the image to binary (you may need to adjust the threshold)
    # If img_ref exists, use it to find the centroid
    if img_ref is None:
        h, w = img_test.shape
        cX_new, cY_new = w // 2, h // 2  # Center of the image
    else:
        backgroundRef = stats.mode(img_ref.flatten())[0]
        binary = img_ref != backgroundRef
        binary = binary.astype(np.uint8) * 255  # Convert boolean to uint8 for display

        # Calculate moments
        M = cv2.moments(binary)

        if M["m00"] != 0:
            cX_new = int(M["m10"] / M["m00"])
            cY_new = int(M["m01"] / M["m00"])
        else:
            raise ValueError("The reference image is empty or has no foreground pixels.")

    # Calculate the translation needed to center the image
    h, w = img_test.shape
    translation_x = cX_new - cX_old
    translation_y = cY_new - cY_old
    # Create a translation matrix
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    # Apply the translation to the test image
    img_test_centered = cv2.warpAffine(img_test, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=int(backgroundTest))

    return img_test_centered

def scale_image(img_test: np.ndarray, img_ref: np.ndarray) -> np.ndarray:

    # Determine background value by taking mode of pixel values and binarize
    backgroundVal = stats.mode(img_ref.flatten())[0]
    binary = img_ref != backgroundVal
    binary = binary.astype(np.uint8) * 255  # Convert boolean to uint8 for display

    # Calculate bounding box of the foreground in the reference image
    _, _, w, h = cv2.boundingRect(binary)
    # Calculate largest dimension
    largest_dim = max(w, h)

    # Calculate bounding box of the foreground in the test image
    backgroundVal = stats.mode(img_test.flatten())[0]
    binary = img_test != backgroundVal
    binary = binary.astype(np.uint8) * 255  # Convert boolean to uint8 for display
    _, _, w_test, h_test = cv2.boundingRect(binary)
    largest_dim_test = max(w_test, h_test)

    # Calculate scaling factor to scale the test image to match the largest dimension of the reference image
    scale_factor = largest_dim / largest_dim_test

    # Scale the test image
    img_test_scaled = cv2.resize(img_test, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Crop the scaled test image to match the reference image size,
    # but adjust the crop to include the entire object (foreground).

    h_ref, w_ref = img_ref.shape
    h_test_scaled, w_test_scaled = img_test_scaled.shape

    # Find the bounding box of the foreground in the scaled test image
    backgroundVal_scaled = stats.mode(img_test_scaled.flatten())[0]
    binary_scaled = img_test_scaled != backgroundVal_scaled
    binary_scaled = binary_scaled.astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(binary_scaled)

    # Find the x and y limits of the object in the scaled image
    obj_indices = np.argwhere(binary_scaled)
    y_obj_min, x_obj_min = obj_indices.min(axis=0)
    y_obj_max, x_obj_max = obj_indices.max(axis=0)

    # Compute crop bounds based on original and scaled image sizes for center cropping
    crop_x1 = max(0, (w_test_scaled - w_ref) // 2)
    crop_y1 = max(0, (h_test_scaled - h_ref) // 2)
    crop_x2 = crop_x1 + w_ref
    crop_y2 = crop_y1 + h_ref

    # If the object's right edge exceeds the crop box, shift the crop box right
    excess_right = x_obj_max - (crop_x2 - 1)
    if excess_right > 0:
        shift = min(excess_right, w_test_scaled - crop_x2)
        crop_x1 += shift
        crop_x2 += shift

    # If the object's left edge is left of the crop box, shift left
    excess_left = crop_x1 - x_obj_min
    if excess_left > 0:
        shift = min(excess_left, crop_x1)
        crop_x1 -= shift
        crop_x2 -= shift

    # If the object's bottom edge exceeds the crop box, shift the crop box down
    excess_bottom = y_obj_max - (crop_y2 - 1)
    if excess_bottom > 0:
        shift = min(excess_bottom, h_test_scaled - crop_y2)
        crop_y1 += shift
        crop_y2 += shift
    
    # If the object's top edge is above the crop box, shift up
    excess_top = crop_y1 - y_obj_min
    if excess_top > 0:
        shift = min(excess_top, crop_y1)
        crop_y1 -= shift
        crop_y2 -= shift

    # If the scaled image is larger, crop to the computed bounds
    if h_test_scaled > h_ref or w_test_scaled > w_ref:
        # Ensure crop bounds are within image dimensions
        crop_x1 = max(0, min(crop_x1, w_test_scaled - w_ref))
        crop_x2 = min(w_test_scaled, crop_x2)
        crop_y1 = max(0, min(crop_y1, h_test_scaled - h_ref))
        crop_y2 = min(h_test_scaled, crop_y2)
        img_test_scaled = img_test_scaled[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        # If the scaled image is smaller, pad it to match the reference size
        pad_top = max(0, (h_ref - h_test_scaled) // 2)
        pad_bottom = max(0, h_ref - h_test_scaled - pad_top)
        pad_left = max(0, (w_ref - w_test_scaled) // 2)
        pad_right = max(0, w_ref - w_test_scaled - pad_left)
        img_test_scaled = cv2.copyMakeBorder(
            img_test_scaled,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=int(backgroundVal)
        )

    return img_test_scaled

def rotate_image(img_test: np.ndarray, img_ref: np.ndarray) -> np.ndarray:

    # Determine background value by taking mode of pixel values and binarize
    backgroundVal = stats.mode(img_ref.flatten())[0]
    binary = img_ref != backgroundVal
    binary = binary.astype(np.uint8) * 255

    labeled_image = label(binary)
    props = regionprops(labeled_image)

    if not props:
        print("No objects found in the image.")
        return None, None
    
    # Assuming the largest object is the one of interest
    largest_object = max(props, key=lambda p: p.area)
    
    # The 'orientation' property gives the angle in radians
    orient_ref = largest_object.orientation
    # print(f"Orientation (radians): {orient_ref}")

    # Determine background value by taking mode of pixel values and binarize
    backgroundVal = stats.mode(img_test.flatten())[0]
    binary = img_test != backgroundVal
    binary = binary.astype(np.uint8) * 255
    height, width = img_test.shape[:2]

    labeled_image = label(binary)
    props = regionprops(labeled_image)

    if not props:
        print("No objects found in the image.")
        return None, None
    
    # Assuming the largest object is the one of interest
    largest_object = max(props, key=lambda p: p.area)
    
    # The 'orientation' property gives the angle in radians
    orient_test = largest_object.orientation
    # print(f"Orientation (radians): {orient_test}")

    # Get transformation matrix for rotation
    rot_angle_deg = (orient_ref - orient_test)*180/np.pi  # Convert radians to degrees
    # print(orient_ref, orient_test, rot_angle_deg)
    rot_mat = cv2.getRotationMatrix2D((width // 2, height // 2), rot_angle_deg, 1.0)
    img_test_rotated = cv2.warpAffine(img_test, rot_mat, (width, height),borderMode=cv2.BORDER_CONSTANT, borderValue=int(backgroundVal))

    img_rotated_placed = center_image(img_test_rotated, img_test)

    return img_rotated_placed

def texture_inplace(img_test: np.ndarray, n_scales: int = 2, max_iter: int = 1500, device: str = 'auto') -> np.ndarray:

    """
    Generate texture metamer of the reference image.
    
    Parameters:
    -----------
    img_test : numpy array
        Image to perform operation on
    n_scales : int, default=2
        Number of scales for Portilla-Simoncelli model
    max_iter : int, default=1500
        Maximum iterations for synthesis
    device : str, default='auto'
        'cuda', 'cpu', or 'auto' (auto-detect)
    """

    plt.rcParams['animation.html'] = 'html5'
    # use single-threaded ffmpeg for animation writer
    plt.rcParams['animation.writer'] = 'ffmpeg'
    plt.rcParams['animation.ffmpeg_args'] = ['-threads', '1']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(device)
        
    if DEVICE.type == 'cuda':
        print("Running on GPU!")
    else:
        print("Running on CPU!")

    # Compute bounding box of the foreground in the reference image and crop it
    backgroundVal = stats.mode(img_test.flatten())[0]
    binary = img_test != backgroundVal
    binary = binary.astype(np.uint8) * 255  # Convert boolean to uint8 for display
    x, y, w, h = cv2.boundingRect(binary)
    img_height, img_width = img_test.shape[:2]

    # print(h,w)
    # Adjust cropping size to ensure its shape can be divided by 2 `n_scales` times
    target_h = ((h + 3) // 4) * 4  # Round up to nearest multiple of 4
    target_w = ((w + 3) // 4) * 4  # Round up to nearest multiple of 4

    # Ensure minimum size of 64x64
    target_h = max(target_h, 64)
    target_w = max(target_w, 64)

    # Simple bounds checking - if target size exceeds image, adjust position
    if x + target_w > img_width:
        x = max(0, img_width - target_w)
    if y + target_h > img_height:
        y = max(0, img_height - target_h)
    
    # If image is still too small, crop what we can and pad the rest
    actual_h = min(target_h, img_height - y)
    actual_w = min(target_w, img_width - x)
    
    print(f"Final crop: x={x}, y={y}, size={actual_h}x{actual_w}, target={target_h}x{target_w}")

    # Crop the reference image
    img_crop = img_test[y:y+actual_h, x:x+actual_w]

    # Pad if necessary to reach target dimensions
    pad_h = target_h - actual_h
    pad_w = target_w - actual_w
    
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        img_crop = np.pad(
            img_crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=int(backgroundVal)
        )
        print(f"Added padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")

    # Convert to tensor and normalize
    img_crop = np.expand_dims(img_crop, axis=0)  # Add batch dimension
    img_crop = np.expand_dims(img_crop, axis=0)  # Add channel dimension
    img_crop = torch.tensor(img_crop, dtype=torch.float32)
    if img_crop.max() > 1.0:
        img_crop /= 255.0

    img_crop = img_crop.to(DEVICE)
    print(f"Image shape after processing: {img_crop.shape}")

    # Create Portilla-Simoncelli model with 2 scales
    ps = po.simul.PortillaSimoncelli(img_crop.shape[-2:], n_scales=n_scales)
    ps.to(DEVICE)

    im_init = torch.rand_like(img_crop) * 0.2 + img_crop.mean()
    met = po.synth.MetamerCTF(img_crop, ps, loss_function=po.tools.optim.l2_norm)
    met.setup(im_init)
    met.synthesize(max_iter=max_iter, store_progress = 10,
               change_scale_criterion=None,
               ctf_iters_to_check=3)
    
    img_crop = po.to_numpy(met.metamer).squeeze()
    new_h, new_w = img_crop.shape

    # Place the result back into the original image
    img_test_crop = np.full_like(img_test, np.mean(img_crop.flatten()) * 255)
    
    # Ensure bounds when placing back
    end_y = min(y + new_h, img_test_crop.shape[0])
    end_x = min(x + new_w, img_test_crop.shape[1])
    crop_h = end_y - y
    crop_w = end_x - x
    
    img_test_crop[y:end_y, x:end_x] = img_crop[:crop_h, :crop_w] * 255

    return img_test_crop

def texture_crop(img_test: np.ndarray, n_scales: int = 2, target_size: int = 256, max_iter: int = 1500, device: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:

    """
    Generate texture metamer with automatic cropping.
    
    Parameters:
    -----------
    img_test : numpy array
        Image to perform operation on
    n_scales : int, default=2
        Number of scales for Portilla-Simoncelli model
    target_size : int, default=256
        Size to resize cropped image
    max_iter : int, default=1500
        Maximum iterations for synthesis
    device : str, default='auto'
        'cuda', 'cpu', or 'auto' (auto-detect)
    """

    plt.rcParams['animation.html'] = 'html5'
    # use single-threaded ffmpeg for animation writer
    plt.rcParams['animation.writer'] = 'ffmpeg'
    plt.rcParams['animation.ffmpeg_args'] = ['-threads', '1']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE.type == 'cuda':
        print("Running on GPU!")
    else:
        print("Running on CPU!")

    # Preprocessing - reduce noise
    img_blur = cv2.GaussianBlur(img_test, (5, 5), 0)

    # Adaptive thresholding - good for non-uniform backgrounds
    binary1 = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 4  # Larger block size, higher C
    )

    _, binary2 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Edge detection to catch faint edges
    edges = cv2.Canny(img_blur, 30, 100)

    # Combine all approaches
    binary = cv2.bitwise_or(binary1, binary2)
    binary = cv2.bitwise_or(binary, edges)

    # Morphological operations to connect components
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)

    # Fill holes
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel_small)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50  # adjust based on your image size
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if len(valid_contours) > 1:
        # Find bounding box that encompasses all significant contours
        all_points = np.vstack([contour.reshape(-1, 2) for contour in valid_contours])
        x, y, w, h = cv2.boundingRect(all_points)
    else:
        # Single contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

    # Find whether width or height is larger then adjust the other dimension to make it square then resize to 256x256
    if w > h:
        diff = w - h
        y = max(0, y - diff // 2)
        h = w
        if y + h > img_test.shape[0]:
            h = img_test.shape[0] - y
    else:
        diff = h - w
        x = max(0, x - diff // 2)
        w = h
        if x + w > img_test.shape[1]:
            w = img_test.shape[1] - x

    # Crop the reference image
    img_crop = img_test[y:y+h, x:x+w]

    # Resize the cropped image to target size
    img_crop = cv2.resize(img_crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    h, w = img_crop.shape

    # Convert the reference image to a tensor and move it to the appropriate device
    img_prep = np.expand_dims(img_crop, axis=0)  # Add batch dimension
    img_prep = np.expand_dims(img_prep, axis=0)  # Add channel dimension
    img_prep = torch.tensor(img_prep, dtype=torch.float32)
    if img_prep.max() > 1.0:
        img_prep /= 255.0  # Normalize to [0, 1] if necessary

    img_prep = img_prep.to(DEVICE)

    # print(f"Image shape after cropping: {img_crop.shape}")
    ps = po.simul.PortillaSimoncelli(img_prep.shape[-2:],n_scales=n_scales)
    ps.to(DEVICE)

    im_init = torch.rand_like(img_prep) * 0.2 + img_prep.mean()
    met = po.synth.MetamerCTF(img_prep, ps, loss_function=po.tools.optim.l2_norm, coarse_to_fine='together')
    met.setup(im_init)
    met.synthesize(max_iter=max_iter, store_progress = 10,
               change_scale_criterion=None,
               ctf_iters_to_check=3)
    
    img_synth = po.to_numpy(met.metamer).squeeze() * 255

    return img_crop, img_synth

def skeletonize_object(img_test: np.ndarray, area_threshold: int = 3, blur_kernel: Tuple[int, int] = (5, 5), invert: bool = True) -> Tuple[np.ndarray, np.ndarray]:

    """
    Skeletonize an object in the image.
    
    Parameters:
    -----------
    img_ref : numpy array
        Reference image
    area_threshold : int, default=3
        Area threshold for removing small holes
    blur_kernel : tuple, default=(5, 5)
        Kernel size for Gaussian blur
    invert : bool, default=True
        Whether to invert colors at the end
    """

    # Convert the image to grayscale if it is not already
    if len(img_test.shape) == 3:
        img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image and fill small holes
    backgroundVal = stats.mode(img_test.flatten())[0]
    binary = img_test != backgroundVal
    filled_mask = remove_small_holes(binary, area_threshold=area_threshold)
    closed_img = cv2.morphologyEx(filled_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel=np.ones((3,3),np.uint8))

    # Skeletonize the binary image
    skeleton = skeletonize(closed_img)

    # Convert skeleton back to uint8 for display
    skeleton = (skeleton * 255).astype(np.uint8)

    # Use Gaussian blur to smooth the skeleton
    skeleton = cv2.GaussianBlur(skeleton, blur_kernel, 0)

    # Invert colors to match original image style (object white on black background)
    if invert:
        skeleton = cv2.bitwise_not(skeleton)

    return closed_img * 255, skeleton

def NN_activation(img_test: np.ndarray, network: str = 'alexnet', layer_types: Optional[List[str]] = None, device: str = 'auto') -> List[np.ndarray]:

    """
    Extract neural network activations from specified layers.
    
    Parameters:
    -----------
    img_ref : numpy array
        Reference image
    network : str, default='alexnet'
        Network architecture ('alexnet', 'vgg16', 'resnet50', etc.)
    layer_types : list, default=None
        Layer types to extract (e.g., ['Conv2d', 'Linear']). 
        If None, defaults to ['Conv2d', 'Linear']
    device : str, default='auto'
        'cuda', 'cpu', or 'auto' (auto-detect)
    """

    if layer_types is None:
        layer_types = ['Conv2d', 'Linear']

    activations = {}

    def register_hook_for_layer(net, layer_name):
        def hook(module, input, output):
            activations[layer_name] = output.detach().cpu().numpy()
        layer = dict(net.named_modules())[layer_name]
        layer.register_forward_hook(hook)

    def load_pretrained_model(model_name):
        model_path = MODEL_PATHS[model_name]
        assert os.path.isfile(model_path)

        model, layers = load_model(
            model_name, 
            trained=True, 
            model_path=model_path, 
            model_family="imagenet",
            state_dict_key="model_state_dict",  # make sure `model_state_dict` is in the *.pt file
        )

        return model, layers

    # Load the specified network
    if network == 'alexnet':
        net = models.alexnet(pretrained=True)
    elif network == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif network == 'resnet50':
        net = models.resnet50(pretrained=True)
    elif network == 'vgg11':
        net = models.vgg11(pretrained=True)
    elif network == 'resnet18':
        net = models.resnet18(pretrained=True)
    elif network == 'alexnet_mouse':
        name = "alexnet_bn_ir_64x64_input_pool_6"
        # load_pretrained_model returns (model, layers) to match the project's
        # internal loader convention. Unpack so `net` is the model itself —
        # this keeps behavior consistent with the torchvision models above.
        net, _ = load_pretrained_model(name)
    else:
        raise ValueError(f"Unsupported network: {network}")

    # Move the network to the selected device so its weights and buffers
    # have the same device type as the input tensor we will send through it.
    if device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(device)
    net = net.to(DEVICE)

    layer_names = []
    layers = []
    # build a lookup of all modules upfront
    all_modules = {name: layer for name, layer in net.named_modules()}
    all_types = {layer.__class__.__name__ for layer in all_modules.values()}

    if isinstance(layer_types, str):
        layer_types = [layer_types]

    for entry in layer_types:
        entry = entry.strip()
        has_type = entry.split(':')[0].strip() in all_types if ':' in entry else entry in all_types
        has_name = entry.split(':')[-1].strip() in all_modules if ':' in entry else entry in all_modules

        if has_type and has_name:                      # "ReLU: features.2" — exact match
            target_type, target_name = [s.strip() for s in entry.split(':')]
            layer_names.append(f'{target_type}: {target_name}')
            layers.append(target_name)

        elif has_type:                                  # "ReLU" — all layers of that type
            target_type = entry
            for name, layer in all_modules.items():
                if layer.__class__.__name__ == target_type:
                    layer_names.append(f'{target_type}: {name}')
                    layers.append(name)

        elif has_name:                                  # "features.2" — that specific layer
            layer = all_modules[entry]
            layer_names.append(f'{layer.__class__.__name__}: {entry}')
            layers.append(entry)

        else:
            raise ValueError(f"'{entry}' is not a recognized layer type or name")

    DNN_input = torch.tensor(img_test, dtype=torch.float32).to(DEVICE)  # Ensure DNN_input is of type Float
    if len(DNN_input.shape) == 2:  # If grayscale, add channel dimension and triple to 3 channels
        DNN_input = DNN_input.unsqueeze(2).repeat(1, 1, 3)
    if len(DNN_input.shape) == 3:  # If single image, add batch dimension
        DNN_input = DNN_input.unsqueeze(3)
    DNN_input = DNN_input.permute(3, 2, 0, 1)

    features = []  # Preallocate feature array
    # print(layers)
    for i, layer_name in enumerate(layers):
        # print(f"Extracting features from layer {i+1}/{len(layers)}: {layer_name}")
        # Extract features at specific layer
        register_hook_for_layer(net, layer_name)

        net(DNN_input)  # Forward pass to trigger hooks
        features.append(activations[layer_name])

    return features, layer_names