import cv2
import numpy as np
import time

class DisjointSet:
    def __init__(self, size):
        self.parent = [-1] * size

    def find(self, n, var="default"):
        """
        Function Find returns root of node, n in O(n) worst case time
        :param n: integer (node)
        :param var: string (variation of find to use; default or path compression)
        :return: integer (root of node)
        """
        if self.parent[n] < 0:  # return node itself as root if parent[n] is negative value
            return n
        else:  # return root of node
            if var == "path_compression":  # condition to apply path compression
                root = self.find(self.parent[n], "path_compression")  # recursively calls find until root node is found
                self.parent[n] = root  # sets parent of n to root for each recursive call
                return root
            else:  # default find
                return self.find(self.parent[n])  # recursively calls find until root node is found

    def union(self, n1, n2, uvar="default", fvar="default"):
        """
        Function Union joins nodes if they are not in the same set
        and returns True if a union occurs and False otherwise.
        O(n) worst case time based on Find()
        :param n1: integer (node 1)
        :param n2: integer (node 2)
        :param uvar: string (variation of Union to use; size or rank)
        :param fvar: string (variation of Find to use; default or path compression)
        :return: boolean
        """
        root_n1 = self.find(n1, fvar)  # find root of n1
        root_n2 = self.find(n2, fvar)  # find root of n2
        if root_n1 == root_n2:  # part of same set
            return False
        val_n1 = -self.parent[root_n1]  # get size/rank value of n1
        val_n2 = -self.parent[root_n2]  # get size/rank value of n2
        if val_n1 < val_n2:
            self.parent[root_n1] = root_n2  # assigning new parent
            root = root_n2
        else:
            self.parent[root_n2] = root_n1  # assigning new parent
            root = root_n1
        if uvar == "size":
            self.parent[root] = -(val_n1 + val_n2)  # recalculating size
        elif uvar == "rank" and val_n1 == val_n2:
            self.parent[root] = self.parent[root] - 1  # recalculating rank
        return True


def connected_components(image):
    """
    Takes a grayscale im and counts component connectivity returning a histogram containing number of components
    that are within 50% size of the largest component.
    :param image: grayscale image
    :return: histogram, vals = [100-90, 90-80, 80-70, 70-60, 60-50]
    """
    (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    height, width = im_bw.shape

    # Identify background colour
    background_check = 0
    for y in range(height):
        if im_bw[y][0] == 0:
            background_check += 1
        else:
            background_check += -1
        if im_bw[y][width-1] == 0:
            background_check += 1
        else:
            background_check += -1
    for x in range(width):
        if im_bw[0][x] == 0:
            background_check += 1
        else:
            background_check += -1
        if im_bw[height - 1][x] == 0:
            background_check += 1
        else:
            background_check -= 1
    if background_check > 0:  # background is black then flip
        im_bw = cv2.bitwise_not(im_bw)

    # Increase connectivity of components
    strelsize = int((height + width) / 200)
    kernel = np.ones((strelsize, strelsize), np.uint8)
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)  # open image to increase connectivity

    im_bw_boundary_add = np.zeros((height + 2, width + 2))
    im_bw_boundary_add.fill(255)
    im_bw_boundary_add[1:height + 1, 1:width + 1] = im_bw
    im_bw = im_bw_boundary_add

    # Getting components
    black_components = DisjointSet((height + 2) * (width + 2))
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            if im_bw[y][x] == 0:
                    if im_bw[y + 1][x + 1] == 0:
                        black_components.union((y * width) + x, ((y + 1) * width) + (x + 1), uvar="size", fvar="path_compression")
                    if im_bw[y + 1][x - 1] == 0:
                        black_components.union((y * width) + x, ((y + 1) * width) + (x - 1), uvar="size", fvar="path_compression")
                    if im_bw[y + 1][x] == 0:
                        black_components.union((y * width) + x, ((y + 1) * width) + x, uvar="size", fvar="path_compression")
                    if im_bw[y][x + 1] == 0:
                        black_components.union((y * width) + x, (y * width) + (x + 1), uvar="size", fvar="path_compression")
                    if im_bw[y][x - 1] == 0:
                        black_components.union((y * width) + x, (y * width) + (x - 1), uvar="size", fvar="path_compression")
                    if im_bw[y - 1][x + 1] == 0:
                        black_components.union((y * width) + x, ((y - 1) * width) + (x + 1), uvar="size", fvar="path_compression")
                    if im_bw[y - 1][x - 1] == 0:
                        black_components.union((y * width) + x, ((y - 1) * width) + (x - 1), uvar="size", fvar="path_compression")
                    if im_bw[y - 1][x] == 0:
                        black_components.union((y * width) + x, ((y - 1) * width) + x, uvar= "size", fvar="path_compression")
    components = [None] * ((height + 2) * (width + 2))
    # Getting bounding box
    # Scan top down
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            pos = (y * width) + x
            root = black_components.find(pos, var="path_compression")
            if black_components.parent[root] < -1:
                if components[root] is None:
                    components[root] = [y, None, None, None]  # [top, bottom, left, right]
    # Scan left to right
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            pos = (y * width) + x
            root = black_components.find(pos, var="path_compression")
            if black_components.parent[root] < -1:
                if components[root][2] is None:
                    components[root][2] = x
    # Scan right to left
    for x in range(width, 0, -1):
        for y in range(1, height + 1):
            pos = (y * width) + x
            root = black_components.find(pos, var="path_compression")
            if black_components.parent[root] < -1:
                if components[root][3] is None:
                    components[root][3] = x
    # Scan bottom up
    for y in range(height, 0, -1):
        for x in range(1, width + 1):
            pos = (y * width) + x
            root = black_components.find(pos, var="path_compression")
            if black_components.parent[root] < -1:
                if components[root][1] is None:
                    components[root][1] = y

    # Fill bounding box
    im_bw = np.zeros((height + 2, width + 2))
    im_bw.fill(255)
    count = 0
    for i in range(len(components)):
        if components[i] is not None:
            count += 1
            y1 = components[i][0]
            y2 = components[i][1]
            x1 = components[i][2]
            x2 = components[i][3]
            im_bw[y1:y2+1, x1:x2+1] = 0

    # Get new sizes
    black_components = DisjointSet((height + 2) * (width + 2))
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            if im_bw[y][x] == 0:
                if im_bw[y + 1][x + 1] == 0:
                    black_components.union((y * width) + x, ((y + 1) * width) + (x + 1), uvar="size", fvar="path_compression")
                if im_bw[y + 1][x - 1] == 0:
                    black_components.union((y * width) + x, ((y + 1) * width) + (x - 1), uvar="size", fvar="path_compression")
                if im_bw[y + 1][x] == 0:
                    black_components.union((y * width) + x, ((y + 1) * width) + x, uvar="size", fvar="path_compression")
                if im_bw[y][x + 1] == 0:
                    black_components.union((y * width) + x, (y * width) + (x + 1), uvar="size", fvar="path_compression")
                if im_bw[y][x - 1] == 0:
                    black_components.union((y * width) + x, (y * width) + (x - 1), uvar="size", fvar="path_compression")
                if im_bw[y - 1][x + 1] == 0:
                    black_components.union((y * width) + x, ((y - 1) * width) + (x + 1), uvar="size", fvar="path_compression")
                if im_bw[y - 1][x - 1] == 0:
                    black_components.union((y * width) + x, ((y - 1) * width) + (x - 1), uvar="size")
                if im_bw[y - 1][x] == 0:
                    black_components.union((y * width) + x, ((y - 1) * width) + x, uvar="size")

    # Keep components that are at least 5% size to the whole image size
    components = []
    for i in range(len(black_components.parent)):
        if -black_components.parent[i] >= height * width * 0.05:
            components.append(int(-black_components.parent[i]))

    largest = max(components)
    n = 0
    for i in range(len(components)):
        if components[i] >= largest//(2*height*width):
            n += 1

    if n >= 2:
        return True
    else:
        return False

