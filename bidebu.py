from ultralytics import YOLO
import cv2
import numpy as np
from algorithms.landing import check_landing_conditions
#model yükle
model = YOLO("models/bu.pt")

#sınıf filtreleme
def map_label(original_label):
    if original_label in ["uap", "uai", "insan"]:
        return original_label
    else:
        return "tasit"

#video path
input_video = "C:/Users/safiy/PycharmProjects/teknofest_uyz2025/test_videos/uapoverlap.mp4"
output_video = "C:/Users/safiy/PycharmProjects/teknofest_uyz2025/test_videos/ouputs/output.mp4"

#video ayarları
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
resize_width = 960
resize_height = 720

# IOU hesaplama fonksiyonu
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union != 0 else 0


# Video döngüsü
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    boxes = results.boxes.data.cpu().numpy()
    selected_boxes = []
    used = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, x2, y2, conf, cls = boxes[i]
        max_conf = conf
        best_box = boxes[i]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            iou = calculate_iou(boxes[i][:4], boxes[j][:4])
            if iou > 0.5:
                if boxes[j][4] > max_conf:
                    best_box = boxes[j]
                    max_conf = boxes[j][4]
                used[j] = True
        selected_boxes.append(best_box)

    detections = []
    for box in selected_boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls = int(cls)
        original_label = model.names[cls]
        filtered_label = map_label(original_label)  # <-- burada filtreleme yapılıyor

        detections.append({'label': filtered_label, 'bbox': (x1, y1, x2, y2)})

        # Renk seçimi: farklı sınıflara göre
        color = (0, 255, 0)
        if filtered_label == "insan":
            color = (255, 255, 0)
        elif filtered_label == "uap":
            color = (0, 0, 255)
        elif filtered_label == "uai":
            color = (255, 0, 255)
        elif filtered_label == "tasit":
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{filtered_label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    #iniş kontrol
    statuses = check_landing_conditions(detections)
    for (x1, y1, x2, y2), msg in [(s[0], s[1]) for s in statuses]:
        cv2.putText(frame, msg, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if "UYGUN DEGIL" in msg else (0, 255, 0), 2)

    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    out.write(frame)
    cv2.imshow("Object Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()