import logging
import time
import random
import requests
from ultralytics import YOLO
from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation

import cv2
import os
from .get_ratioo import get_blue_white_ratio
from .constants import landing_statuses
from .landingg import check_landing_conditions


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        # Modelinizi bu kısımda init edebilirsiniz.
        self.model = YOLO(r"C:\Users\safiy\havacilikta-yapay-zeka-yarismasi\models\mehmet.pt")  # Örnektir!

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        # Indirmek istedigimiz frame frames.json dosyasinda mevcut mu kontrol edelim
        image_name = img_url.split("/")[-1]
        # Eger indirecegimiz frame'i daha once indirmediysek indirme islemine gecelim
        if image_name not in images_files:
            for attempt in range(retries):
                try:
                    response = requests.get(img_url, timeout=60)
                    response.raise_for_status()

                    img_bytes = response.content
                    with open(images_folder + image_name, 'wb') as img_file:
                        img_file.write(img_bytes)

                    t2 = time.perf_counter()
                    logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
                    return

                except requests.exceptions.RequestException as e:
                    logging.error(f"Download failed for {img_url} on attempt {attempt + 1}: {e}")
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2

            logging.error(f"Failed to download image from {img_url} after {retries} attempts.")
        # Eger indirecegimiz frame'i daha once indirdiysek indirme yapmadan devam edebiliriz
        else:
            logging.info(f'{image_name} already exists in {images_folder}, skipping download.')

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files, frame):
        # Doğru URL'yi oluştur
        if prediction.image_url.startswith("http"):
            image_url = prediction.image_url
        else:
            image_url = evaluation_server_url.rstrip("/") + "/media/" + prediction.image_url.lstrip("/")

        # Görüntüyü indir
        self.download_image(image_url, images_folder, images_files)
        frame=frame
        # Nesne tespiti işlemi
        frame_results = self.detect(prediction, health_status, frame)

        return frame_results

    def detect(self, prediction, health_status, frame):
        results = self.model.predict(source=frame['image_url'], save=False, verbose=False)
        frame=frame
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return prediction  # hiç tespit yoksa

        boxes = results[0].boxes.xyxy.cpu().numpy()  # numpy format
        classes = results[0].boxes.cls.cpu().numpy()
        detections = []
        for idx in range(len(boxes)):
            cls_id = int(classes[idx])
            class_name = self.model.names[cls_id]
            objects=[class_name,boxes[idx]]
            print(objects)
            detections.append(objects)

        print(detections)
        for idx in range(len(boxes)):
            class_id = classes[idx]
            class_name = self.model.names[class_id]  # class_name artık her nesne için alınır

            uap_or_uai_exists = False

            if class_name in ['gemi', 'ikiteker', 'ismakinesi', 'kamyon', 'otobus', 'otomobil', 'scooter', 'tekne',
                              'vagon']:
                cls = 0
            elif class_name == "insan":
                cls = 1
            elif class_name == "uai":
                cls = 3
                uap_or_uai_exists = True
            else:
                cls = 2
                uap_or_uai_exists = True

            if cls not in [2, 3]:
                landing_status = "-1"
            else:
                # detection'lar frame için bir kerede hesaplanmalı, her kutu için değil
                # o yüzden bu fonksiyon döngü dışında 1 kez çağrılmalı, istersen yukarı alabilirim
                landing_status = check_landing_conditions(detections, frame["image_url"])

            top_left_x = boxes[idx][0]
            top_left_y =boxes[idx][1]
            bottom_right_x = boxes[idx][2]
            bottom_right_y = boxes[idx][3]

            print("cls:", cls)

            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            prediction.add_detected_object(d_obj)

        # Health Status işlemi
        if health_status == '0':
            pred_translation_x = random.randint(1, 10)
            pred_translation_y = random.randint(1, 10)
        else:
            pred_translation_x = prediction.gt_translation_x
            pred_translation_y = prediction.gt_translation_y

        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y)
        prediction.add_translation_object(trans_obj)

        return prediction
