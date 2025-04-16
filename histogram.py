import cv2
import numpy as np
import os
import csv
from ultralytics import YOLO
from glob import glob

image_dir = "C:/Users/atala/source/github/sahi_teknofest_2025/dataset"
model_path = "C:/Users/atala/source/github/sahi_teknofest_2025/models/best.pt"
csv_output_path = "blue_ratios.csv"

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

    blue_ratio = blue_pixels / total
    white_ratio = white_pixels / total
    return blue_ratio, white_ratio

model = YOLO(model_path)

def process_images_and_save_ratios(image_dir, model_path, csv_output_path):
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_paths.extend(glob(os.path.join(image_dir, ext)))

    uap_id = 13  # <- senin uap class ID'n
    valid_ids = [uap_id]

    ratios = []

    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "blue_ratio", "white_ratio"])

        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                continue

            results = model(img)[0]
            img_name = os.path.basename(image_path)
            found = False

            for box in results.boxes:
                cls_id = int(box.cls)

                if cls_id in valid_ids:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img[y1:y2, x1:x2]

                    blue_ratio, white_ratio = get_blue_white_ratio(crop)
                    writer.writerow([img_name, blue_ratio, white_ratio])
                    ratios.append((blue_ratio, white_ratio))

                    print(f" {img_name}: Blue Ratio = {blue_ratio:.4f}, White Ratio = {white_ratio:.4f}")
                    found = True
                    break

            if not found:
                print(f" {img_name}: uap/uai bulunamadı")

    if ratios:
        avg_blue = sum(r[0] for r in ratios) / len(ratios)
        avg_white = sum(r[1] for r in ratios) / len(ratios)
        print(f"\nOrtalama Mavi Oranı: {avg_blue:.4f}, Ortalama Beyaz Oranı: {avg_white:.4f}")
    else:
        print("\nHiç oran bulunamadı.")

if __name__ == "__main__":
    process_images_and_save_ratios(image_dir, model_path, csv_output_path)
