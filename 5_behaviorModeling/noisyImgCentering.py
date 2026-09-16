import cv2
import helperFuns as helper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats

def center_image(img_test: np.ndarray, img_ref: np.ndarray, perc_jitter: int) -> np.ndarray:
    
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

    # Calculate range of centering jitter based on percent of image size
    jitter_x = int(w * perc_jitter / 100)
    jitter_y = int(h * perc_jitter / 100)

    # Randomly select a jitter value within the specified range
    jitter_x = np.random.randint(-jitter_x, jitter_x + 1)
    jitter_y = np.random.randint(-jitter_y, jitter_y + 1)

    # Update the translation values with jitter
    translation_x += jitter_x
    translation_y += jitter_y

    # Create a translation matrix
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    # Apply the translation to the test image
    img_test_centered = cv2.warpAffine(img_test, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=int(backgroundTest))

    return img_test_centered

projectpath = '/mnt/DataDrive3/emeyer/TreeShrewObjectRecognition/'
taskName = 'Camel_v2_test_nn' #'Camel_v2_test_nn' 'Camel_Rhino_test_nn' , 'Camel_v2_test_nn', 'Camel_background_matrix'
imgPath = f'{projectpath}stimulusSets/{taskName}/original/'

taskName = 'Camel_v2_test_nn' #'Camel_v2_test_nn' 'Camel_Rhino_test_nn' , 'Camel_v2_test_nn', 'Camel_background_matrix'
getDataPath = projectpath + '4_treeShrewBehavior/behaviorData/'
train_targ_idx, train_dist_idx, _, targ_idx, dist_idx = helper.get_train_targdist_data(getDataPath,taskName)
test_targ_idx = np.setdiff1d(targ_idx, train_targ_idx)
test_dist_idx = np.setdiff1d(dist_idx, train_dist_idx)

# Extract images with indexes in train_targ_idx and train_dist_idx
train_targets = []
for t_idx in train_targ_idx:
    train_targets.append(cv2.imread(f'{imgPath}camel_{int(t_idx)}.png', cv2.IMREAD_GRAYSCALE))
train_targets = np.array(train_targets)

test_targets = []
for t_idx in test_targ_idx:
    test_targets.append(cv2.imread(f'{imgPath}camel_{int(t_idx)}.png', cv2.IMREAD_GRAYSCALE))
test_targets = np.array(test_targets)

train_distractors = []
for d_idx in train_dist_idx:
    train_distractors.append(cv2.imread(f'{imgPath}wrench_{int(d_idx)}.png', cv2.IMREAD_GRAYSCALE))
train_distractors = np.array(train_distractors)

test_distractors = []
for d_idx in train_dist_idx:
    test_distractors.append(cv2.imread(f'{imgPath}wrench_{int(d_idx)}.png', cv2.IMREAD_GRAYSCALE))
test_distractors = np.array(test_distractors)

train_set = np.concatenate((train_targets, train_distractors), axis=0)
train_set = train_set.reshape(len(train_set), -1)
test_set = np.concatenate((test_targets, test_distractors), axis=0)
test_set = test_set.reshape(len(test_set), -1)

# Train SVM model using train_set and train_labels
train_labels = np.concatenate((np.ones(len(train_targets)), np.zeros(len(train_distractors))), axis=0)
test_labels = np.concatenate((np.ones(len(test_targets)), np.zeros(len(test_distractors))), axis=0)

Mdl = make_pipeline(StandardScaler(), LinearSVC(random_state=42))
Mdl.fit(train_set, train_labels)
y_pred = Mdl.predict(test_set)

ref_img = cv2.imread(f'{imgPath}camel_0.png', cv2.IMREAD_GRAYSCALE)
df = pd.read_csv(getDataPath + taskName + ".csv")

jitt = [1, 5, 10, 15, 20]

avg_performance = np.full(len(jitt), np.nan)
for idx, jj in enumerate(jitt):
    correct_choice = np.full(len(df), np.nan)
    for index, row in df.iterrows():
        targ = df['T_Expt_ID'][index]
        dist = df['D_Expt_ID'][index]

        targ_img = cv2.imread(f'{imgPath}camel_{int(targ)}.png', cv2.IMREAD_GRAYSCALE)
        dist_img = cv2.imread(f'{imgPath}wrench_{int(dist)}.png', cv2.IMREAD_GRAYSCALE)

        targ_img_centered = center_image(targ_img, ref_img, 20)
        dist_img_centered = center_image(dist_img, ref_img, 20)

        # Extract scores from trained SVM model
        targ_score = Mdl.decision_function(targ_img_centered.reshape(1, -1))
        dist_score = Mdl.decision_function(dist_img_centered.reshape(1, -1))

        # Determine score difference and accuracy based on score difference
        score_diff = targ_score - dist_score
        correct_choice[index] = 1 if score_diff > 0 else 0

    avg_performance[idx] = np.mean(correct_choice)
    print(avg_performance[idx])

plt.figure()
plt.plot(jitt, avg_performance, marker='o')
plt.xlabel('Centering Jitter (%)')
plt.ylabel('Average Performance')
plt.savefig(f'./figures/noisyImgCentering_performance.png', dpi=300)
plt.show()