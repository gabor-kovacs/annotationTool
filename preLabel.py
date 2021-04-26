import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


def convert_prediction_to_1ch(img):
    """
    Converts colored prediction into one channel prediction ready to use with PixelAnnotationTool

    :param img: prediction image (BGR)
    :return: 1 channel label image
    """

    # Freiburg forest mapping from RBG to class id
    rgb_to_label_id = {
        (255, 255, 255): 0,  # Void
        (170, 170, 170): 1,  # Road
        (0, 255, 0): 2,  # Grass
        (102, 102, 51): 3,  # Vegetation
        (0, 60, 0): 3,  # Tree
        (0, 120, 255): 4,  # Sky
        (0, 0, 0): 5,  # Obstacle
    }

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = np.zeros(img.shape[:2], dtype=np.uint8)
    for k, v in rgb_to_label_id.items():
        out[(img == k).all(axis=2)] = v
    return out


# Iterate over all prediction images in folder 'predictions'
pred_image_paths = glob.glob('./predictions/*pred.png')
for pred_image_path in tqdm(pred_image_paths):
    # Read prediction image
    pred_image = cv2.imread(pred_image_path)

    # Extract file index to later name the output
    file_index = (os.path.splitext(os.path.basename(pred_image_path))[0]).split('_')[0]

    # Covert colored prediction to 1 channel
    watershed_image = convert_prediction_to_1ch(pred_image)
    manual_image = watershed_image.copy()

    # Write output on disk with the appropriate filename
    cv2.imwrite('images/' + file_index + '_rgb_mask.png', manual_image)
    cv2.imwrite('images/' + file_index + '_rgb_watershed_mask.png', watershed_image)
