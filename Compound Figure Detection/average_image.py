import numpy as np
import cv2

def average_image(images):
    """
    Initializes empty 600x600 image and gradually adds pixel values of image set into it
    :param images: list of grayscale images
    :return: Average combined image
    """
    avg_img = np.zeros((600, 600), np.float)
    n_im = len(images)
    for i in range(n_im):
        image = images[i]
        resized = cv2.resize(image, (600, 600))
        array_im = np.array(resized, dtype=np.float)
        avg_img = avg_img + array_im/n_im

    avg_img = np.array(np.round(avg_img), dtype=np.uint8)
    return avg_img
