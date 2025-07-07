import logging
import time
import random
import requests
import torch

from ultralytics import YOLO

from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .landingg import check_landing_conditions
import os

from .connection_handler import ConnectionHandler


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        self.connectionHandler = ConnectionHandler(evaluation_server_url)
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        # Modelinizi bu kısımda init edebilirsiniz.
        # self.model = get_keras_model() # Örnektir!
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(r"C:\Users\atala\PycharmProjects\sahi_teknofest_2025\models\mehmet.pt")
        self.model.to(self.device)

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

    def process(self, prediction,evaluation_server_url,health_status, images_folder ,images_files, frame):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Ornek)
        if prediction.image_url.startswith("http"):
            image_url = prediction.image_url
        else:
            image_url = evaluation_server_url.rstrip("/") + "/media/" + prediction.image_url.lstrip("/")

        # Görüntüyü indir
        self.download_image(image_url, images_folder, images_files)
        frame=frame
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti ve pozisyon kestirim modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction,health_status,frame)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction, health_status, frame):
        session_name = self.connectionHandler.get_session_name()
        image_filename = prediction.image_url.split('/')[-1]
        image_path = os.path.join('_images', session_name, image_filename)
        results = self.model.predict(source=image_path, save=False, verbose=False, device=self.device)
        frame = frame
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return prediction  # hiç tespit yoksa

        boxes = results[0].boxes.xyxy.cpu().numpy()  # numpy format
        classes = results[0].boxes.cls.cpu().numpy()
        detections = []
        for idx in range(len(boxes)):
            cls_id = int(classes[idx])
            class_name = self.model.names[cls_id]
            objects = [class_name, boxes[idx]]
            detections.append(objects)
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
            top_left_y = boxes[idx][1]
            bottom_right_x = boxes[idx][2]
            bottom_right_y = boxes[idx][3]


            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            prediction.add_detected_object(d_obj)

        if health_status == '0':
            # Takimlar buraya kendi gelistirdikleri algoritmalarin sonuclarini entegre edebilirler.
            pred_translation_x = random.randint(1, 10) # Ornek olmasi icin rastgele degerler atanmistir takimlar kendi sonuclarini kullanmalidirlar.
            pred_translation_y = random.randint(1, 10) # Ornek olmasi icin rastgele degerler atanmistir takimlar kendi sonuclarini kullanmalidirlar.
            pred_translation_z = random.randint(1, 10) # Ornek olmasi icin rastgele degerler atanmistir takimlar kendi sonuclarini kullanmalidirlar.
        else :
            pred_translation_x = prediction.gt_translation_x # String
            pred_translation_y = prediction.gt_translation_y
            pred_translation_z = prediction.gt_translation_z

        # Translation icin hesaplanilan degerleri sunucuya gondermek icin ilgili objeye dolduralim.
        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y, pred_translation_z)
        prediction.add_translation_object(trans_obj)

        return prediction