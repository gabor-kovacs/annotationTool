import numpy as np
import os
import glob
import cv2
from progress.bar import Bar

void        = [255,255,255]   # id: 0
road        = [170,170,170]   # id: 1
grass       = [0,255,0]       # id: 2
vegetation  = [51,102,102]    # id: 3
sky         = [255,120,0]     # id: 4
obstacle    = [0,0,0]         # id: 5


predictionImagePaths = glob.glob('./predictions/*pred.png')
bar = Bar('Processing', max=len(predictionImagePaths))

for PredictionImagePath in predictionImagePaths:
  PredictionImage = cv2.imread(PredictionImagePath)

  fileIndex = (os.path.splitext(os.path.basename(PredictionImagePath))[0]).split('_pred')[0]

  height = PredictionImage.shape[0]
  width = PredictionImage.shape[1]
  watershedImage = np.zeros((height,width,3), np.uint8)

  for y in range(0, height):
    for x in range(0, width):
      if (PredictionImage[y, x] == void).all():           watershedImage[y, x] = [0,0,0]
      if (PredictionImage[y, x] == road).all():           watershedImage[y, x] = [1,1,1]
      if (PredictionImage[y, x] == grass).all():          watershedImage[y, x] = [2,2,2]
      if (PredictionImage[y, x] == vegetation).all():     watershedImage[y, x] = [3,3,3]
      if (PredictionImage[y, x] == sky).all():            watershedImage[y, x] = [4,4,4]
      if (PredictionImage[y, x] == obstacle).all():       watershedImage[y, x] = [5,5,5]
     
  manualImage = watershedImage.copy()

  cv2.imwrite('images/' + fileIndex + '_rgb_mask.png', manualImage)
  cv2.imwrite('images/' + fileIndex + '_rgb_watershed_mask.png', watershedImage)
  bar.next()
  
bar.finish()
