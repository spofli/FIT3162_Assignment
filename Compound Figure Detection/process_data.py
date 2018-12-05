import os
import cv2
import numpy as np
from CCA import connected_components
from get_lbp_feature import get_lbp_feature
from neighbour_points import neighbour_points
import time

t0 = time.time()

#np.save("image_data5K", im_dataset)
#np.save("class5k", im_classification)
im_dataset = np.load("image_data5K.npy")
im_classification = np.load("class5k.npy")
print()

params = [["nrriu2", 1, [3,2], [2,3]],
          ["riu2", 1, [3,2], [2,3]],
          ["nr", 1, [3,2], [2,3]],
          ["nrriu2", 1, [2], [2]],
          ["riu2", 1, [2], [2]],
          ["nr", 1, [2], [2]],
          ["nrriu2", 1, [3], [3]],
          ["riu2", 1, [3], [3]],
          ["nr", 1, [3], [3]],
          ["nrriu2", 1, [5], [5]],
          ["riu2", 1, [5], [5]],
          ["nr", 1, [5], [5]]]

for p in range(len(params)):
    neighbour_info = neighbour_points(params[p][1])
    print("Processing image data to get feature descriptors")
    # Extract LBP/ CLBP features from each image in dataset
    lbp_features = [None] * len(im_dataset)
    clbp_features = [None] * len(im_dataset)
    for i in range(len(im_dataset)):
        lbp_features[i], clbp_features[i] = get_lbp_feature(im_dataset[i], neighbour_info, 210, 210, params[p][2], params[p][3], maptype=params[p][0],
                                                                radius=params[p][1])
        print(p, i)
    np.save("5kfeatures_lbp_%s_r%d_v%d" % (params[p][0], params[p][1], p), lbp_features)
    np.save("5kfeatures_clbp_%s_r%d_v%d" % (params[p][0], params[p][1], p), clbp_features)
print(time.time() - t0)
print("done")