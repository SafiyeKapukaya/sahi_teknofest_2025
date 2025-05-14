import cv2
import numpy as np
import urllib.request
from .get_ratioo import get_blue_white_ratio, get_red_white_ratio

# URL'den OpenCV görüntüsü yükleme
def url_to_image(url):
    try:
        resp = urllib.request.urlopen(url)
        image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print("Görüntü indirilemedi:", e)
        return None

# Kutu çakışıyor mu kontrolü
def boxes_overlap(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    return x2_min > x1_max and y2_min > y1_max

# İniş uygunluk kontrolü
def check_landing_conditions(detections, frame_url):
    frame = url_to_image(frame_url)
    if frame is None:
        print("Görüntü yüklenemedi.")
        return []

    landing_zones = [det for det in detections if det[0] in ["uap", "uai"]]
    other_objects = [det for det in detections if det[0] not in ["uap", "uai"]]

    status = []
    for zone in landing_zones:
        zone_bbox = zone[1]
        overlap = any(boxes_overlap(zone_bbox, obj[1]) for obj in other_objects)

        x1, y1, x2, y2 = map(int, zone_bbox)
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            print(f"Boş kırpma: {zone_bbox}")
            status.append((zone_bbox, "0"))
            continue

        if zone[0] == "uap":
            ratio_blue = get_blue_white_ratio(cropped)
            if not overlap and 0.13 <= ratio_blue <= 0.27:
                status.append((zone_bbox, "1"))
            else:
                status.append((zone_bbox, "0"))
        else:  # uai
            ratio_red = get_red_white_ratio(cropped)
            if not overlap and 0.06 <= ratio_red <= 0.16:
                status.append((zone_bbox, "1"))
            else:
                status.append((zone_bbox, "0"))
    return status[0][1]
