import cv2
from clbp import clbp
import mapping
import numpy as np


def get_lbp_feature(image, points, new_width, new_height, x_arrange, y_arrange, radius=1, maptype=None):
    """
    Takes an image, mapping type, radius, and configuration to apply clbp
    and returns the clbp and lbp feature histogram for that configuration.
    Configuration examples: [[3,3],[25,25]] means 3 by 3 windows (9 windows in total) where
    size of window is 25x25 pixels.
    Neighbours to sample fixed at 8*radius maxed at 24
    :param image: grayscale image
    :param points: array of 4 arrays containing data to get neighbour point values with and without interpolation
    :param maptype: string, if None, no mapping
    :param new_width: int (preferably divisible with no remainder from x_arrange values)
    :param new_height: int (preferably divisible with no remainder from y_arrange values)
    :param x_arrange: array, configuration to arrange windows
    :param y_arrange: array, configuration to arrange windows
    :param radius: int
    :param config: array, if None, apply LBP on image as whole
    :return: concatenated histogram feature descriptor
    """
    # decide number of sample points based on radius
    if radius == 1:
        neighbours = 8
    elif radius == 2:
        neighbours = 16
    else:
        neighbours = 24

    # get mapping for lbp
    if maptype == "riu2":
        table, bins = mapping.riu2_mapping(neighbours)
    elif maptype == "u2":
        table, bins = mapping.u2_mapping(neighbours)
    elif maptype == "ri":
        table, bins = mapping.ri_mapping(neighbours)
    elif maptype == "nr":
        table, bins = mapping.nr_mapping(neighbours)
    elif maptype == "nrriu2":
        table, bins = mapping.nrriu2_mapping(neighbours)
    else:
        table, bins = None, 2 ** neighbours

    # lbp is performed in a sliding window and the feature histograms are concatenated
    # resize image
    resized = cv2.resize(image, (new_width, new_height))
    lbp_feature = np.array([])
    clbp_feature = np.array([])
    x_size = []
    y_size = []
    for i in range(len(x_arrange)):
        x_size.append(new_width//x_arrange[i])
        y_size.append(new_height//y_arrange[i])
    for i in range(len(y_arrange)):
        for y in range(y_arrange[i]):
            for x in range(x_arrange[i]):
                window_x = x_size[i] * x
                window_y = y_size[i] * y
                window = resized[window_y:window_y + y_size[i], window_x:window_x + x_size[i]]
                clbp_window_s, clbp_window_m, clbp_window_c = clbp(window, neighbours, radius, points, table)
                hist, _ = np.histogram(clbp_window_s, np.arange(bins), density=True)
                lbp_feature = np.concatenate((lbp_feature, hist))
                hist_s, _ = np.histogram(clbp_window_s, np.arange(bins), density=True)
                hist_s = hist_s * 0.75
                hist_m, _ = np.histogram(clbp_window_m, np.arange(bins), density=True)
                hist_m = hist_m * 0.25
                clbp_feature = np.concatenate((clbp_feature, hist_s))
                clbp_feature = np.concatenate((clbp_feature, hist_m))
    return lbp_feature, clbp_feature
