import cv2
import numpy as np
import os
import csv
from glob import glob
def get_blue_white_ratio(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    blue_pixels = cv2.countNonZero(blue_mask)
    white_pixels = cv2.countNonZero(white_mask)
    total = blue_pixels + white_pixels

    if total == 0:
        return 0.0, 0.0

    return white_pixels/blue_pixels

def get_red_white_ratio(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Kırmızı iki aralıkta bulunur: 0-10 ve 160-180
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    red_pixels = cv2.countNonZero(red_mask)
    white_pixels = cv2.countNonZero(white_mask)
    total = red_pixels + white_pixels

    if total == 0:
        return 0.0

    red_ratio = red_pixels / total
    white_ratio = white_pixels / total
    return white_pixels/red_pixels
