import numpy as np


def neighbour_points(R):
    # decide number of sample points based on radius
    if R == 1:
        P = 8
    elif R == 2:
        P = 16
    else:
        P = 24

    # Set angle between points in neighbourhood
    angle = 2 * np.pi / P

    # Get neighbour points relative to central
    neighbours = [None] * P
    interpolation_weights = [None] * P
    neighbours_floor = [None] * P
    neighbours_ceil = [None] * P
    for i in range(P):
        # Getting point position in neighbourhood
        xi = (R * np.cos(i * angle)) + R
        yi = -(R * np.sin(i * angle)) + R
        # Rounding point for list indexing
        round_x = int(np.round(xi))
        round_y = int(np.round(yi))
        # check if interpolation needed (distance of rounded value from original value)
        if abs(xi - round_x) < 10 ** (-6) and abs(yi - round_y) < 10 ** (-6):  # interpolation not needed
            neighbours[i] = [round_y, round_x]
        else:  # interpolation needed
            floor_x = int(np.floor(xi))
            floor_y = int(np.floor(yi))
            ceil_x = int(np.ceil(xi))
            ceil_y = int(np.ceil(yi))
            neighbours_floor[i] = [floor_y, floor_x]
            neighbours_ceil[i] = [ceil_y, ceil_x]
            # interpolation weights
            interpolation_weights[i] = [(1 - (xi - floor_x)) * (1 - (yi - floor_y)), (xi - floor_x) * (1 - (yi - floor_y)), (1 - (xi - floor_x)) * (yi - floor_y), (xi - floor_x) * (yi - floor_y)]
    return [neighbours, neighbours_floor, neighbours_ceil, interpolation_weights]