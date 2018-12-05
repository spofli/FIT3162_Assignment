import numpy as np


def lbp(im, P, R, points, mapping=None):
    """
    Generates the LBP feature for an image
    Common P,R settings are 8-1 (3x3), 16-2 (5x5), 24-3 (7x7)
    :param im: grayscale image
    :param P: number of neighbour points to sample
    :param R: radius from center of points
    :param points: array of 4 arrays containing data to get neighbour point values with and without interpolation
    :param mapping_table: table to remap values (to lower feature dimensionality)
    :return: lbp feature array
    """
    # Get dimensions of image
    height, width = im.shape

    # Get data to calculate neighbour information
    neighbours = points[0]
    neighbours_floor = points[1]
    neighbours_ceil = points[2]
    interpolation_weights = points[3]

    # range to perform LBP
    central_width = width - (R * 2)
    central_height = height - (R * 2)
    central_im = im[R:central_height + R, R:central_width + R]

    # Initialize output image
    if P <= 8:
        lbp_im = np.zeros((central_height, central_width), np.uint8)
    elif P <= 16:
        lbp_im = np.zeros((central_height, central_width), np.uint16)
    else:
        lbp_im = np.zeros((central_height, central_width), np.uint32)

    for i in range(P):
        if neighbours[i] is not None:  # interpolation not needed
            neighbour_val = im[neighbours[i][0]: neighbours[i][0] + central_height,
                            neighbours[i][1]: neighbours[i][1] + central_width]
        else:  # interpolation needed
            # calculate value of interpolated pixel
            im_val0 = interpolation_weights[i][0] * im[neighbours_floor[i][0]: neighbours_floor[i][0] + central_height,
                                                    neighbours_floor[i][1]: neighbours_floor[i][1] + central_width]
            im_val1 = interpolation_weights[i][1] * im[neighbours_floor[i][0]: neighbours_floor[i][0] + central_height,
                                                    neighbours_ceil[i][1]: neighbours_ceil[i][1] + central_width]
            im_val2 = interpolation_weights[i][2] * im[neighbours_ceil[i][0]: neighbours_ceil[i][0] + central_height,
                                                    neighbours_floor[i][1]: neighbours_floor[i][1] + central_width]
            im_val3 = interpolation_weights[i][3] * im[neighbours_ceil[i][0]: neighbours_ceil[i][0] + central_height,
                                                    neighbours_ceil[i][1]: neighbours_ceil[i][1] + central_width]
            neighbour_val = im_val0 + im_val1 + im_val2 + im_val3
        # Get lbp values
        idx = np.where(neighbour_val >= central_im)
        np.add.at(lbp_im, idx, 2 ** i)

    # Check encoding scheme
    if mapping is not None:
        for y in range(central_height):
            for x in range(central_width):
                lbp_im[y][x] = mapping[int(lbp_im[y][x])]

    return lbp_im