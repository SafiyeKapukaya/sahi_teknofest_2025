import cv2
import numpy as np
from .get_ratioo import get_blue_white_ratio,get_red_white_ratio

# kutu çakışıyor mu
def boxes_overlap(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    return x2_min > x1_max and y2_min > y1_max

# iniş uygunluk kontrolü
def check_landing_conditions(detections, frame):
    landing_zones = [det for det in detections if det[0] in ["uap","uai"]]
    other_objects = [det for det in detections if det[0] not in ["uap","uai"]]

    status = []
    for zone in landing_zones:
        zone_bbox = zone[1]
        overlap = any(boxes_overlap(zone_bbox, obj[1]) for obj in other_objects)

        x1, y1, x2, y2 = zone_bbox
        cropped = frame[y1:y2, x1:x2]
        if zone[0] == "uap":
            ratio_blue = get_blue_white_ratio(cropped)
            if not overlap and 0.13 <= ratio_blue <= 0.27:
                status.append((zone_bbox, "1"))
            else:
                status.append((zone_bbox, "0"))
        else:
            ratio_red = get_red_white_ratio(cropped)
            if not overlap and 0.06 <= ratio_red <= 0.16:
                status.append((zone_bbox, "1"))
            else:
                status.append((zone_bbox, "0"))
    return status
