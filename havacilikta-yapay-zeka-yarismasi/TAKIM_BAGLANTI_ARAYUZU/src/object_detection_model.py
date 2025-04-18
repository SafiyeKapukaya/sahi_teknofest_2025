import logging
import time
import os
import cv2
import requests
from ultralytics import YOLO
from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation

class ObjectDetectionModel:
    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        self.model = YOLO("C:/Users/atala/source/github/sahi_teknofest_2025/havacilikta-yapay-zeka-yarismasi/models/best.pt")
        print("objectdet1")

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = os.path.basename(img_url)

        img_path = os.path.join(images_folder, image_name)
        if image_name not in images_files or not os.path.exists(img_path):
            for attempt in range(retries):
                try:
                    response = requests.get(img_url, timeout=60)
                    response.raise_for_status()
                    img_bytes = response.content

                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_bytes)

                    if os.path.getsize(img_path) < 1000:
                        logging.warning(f"Downloaded image may be incomplete: {img_path}")
                    else:
                        logging.info(f'{img_url} - Download finished in {time.perf_counter() - t1:.2f} sec -> {img_path}')
                    return
                except requests.exceptions.RequestException as e:
                    logging.error(f"Download failed ({attempt + 1}/{retries}): {img_url} -> {e}")
                    time.sleep(wait_time)
                    wait_time *= 2
            logging.error(f"Failed to download image after {retries} attempts: {img_url}")
        else:
            logging.info(f'{image_name} already exists, skipping download.')

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files):
        image_url = prediction.image_url
        full_image_url = image_url if image_url.startswith("http") else evaluation_server_url + image_url
        image_name = os.path.basename(full_image_url)

        self.download_image(full_image_url, images_folder, images_files)

        # ✅ Eksik olan satır eklendi
        image_path = os.path.join(images_folder, image_name)

        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            return prediction

        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Image read error (OpenCV failed): {image_path}")
            return prediction

        frame_results = self.detect(prediction, health_status, image_path)
        return frame_results

    def detect(self, prediction, health_status, image_path):
        results = self.model(image_path)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                cls_name = self.model.names[class_id]
                cls = classes.get(cls_name, classes["UAP"])

                top_left_x = int(box.xyxy[0][0])
                top_left_y = int(box.xyxy[0][1])
                bottom_right_x = int(box.xyxy[0][2])
                bottom_right_y = int(box.xyxy[0][3])

                area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
                landing_status = landing_statuses["Inilebilir"] if area < 3000 else landing_statuses["Inilemez"]

                d_obj = DetectedObject(cls, landing_status, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
                prediction.add_detected_object(d_obj)

        # ✅ Her durumda bir değer atanmalı
        pred_translation_x = 0
        pred_translation_y = 0

        if health_status == '0':
            if prediction.detected_objects:
                obj = prediction.detected_objects[0]
                pred_translation_x = (obj.top_left_x + obj.bottom_right_x) // 2
                pred_translation_y = (obj.top_left_y + obj.bottom_right_y) // 2

        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y)
        prediction.add_translation_object(trans_obj)

        return prediction
