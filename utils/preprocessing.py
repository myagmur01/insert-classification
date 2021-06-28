

import os
import cv2
import numpy as np

def mask_generator():
    pass

def contour_detection():
    pass

def hist_equilizer():
    pass

def canny_edge(img_path):

    """ Detect edges in image"""

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 100, 200)
    return img