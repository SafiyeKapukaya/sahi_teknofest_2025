#kutu çakışıyor mu
def boxes_overlap(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    return x2_min > x1_max and y2_min > y1_max

#iniş uygunluk kontrolü
def check_landing_conditions(detections):
    landing_zones = [det for det in detections if det['label'] in ('uap', 'uai')]
    other_objects = [det for det in detections if det['label'] not in ('uap', 'uai')]

    status = []
    for zone in landing_zones:
        zone_label = zone['label']
        zone_bbox = zone['bbox']
        overlap = any(boxes_overlap(zone_bbox, obj['bbox']) for obj in other_objects)
        if overlap:
            status.append((zone_bbox, f"{zone_label.upper()}: UYGUN DEGIL"))
        else:
            status.append((zone_bbox, f"{zone_label.upper()}: UYGUN"))
            return status