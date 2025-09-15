import logging
import time
import requests
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .landingg import check_landing_conditions
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

        self.rgb_model_path = r"C:\Users\atala\PycharmProjects\sahi_teknofest_2025\models\rgb_best.pt"
        self.termal_model_path = r"C:\Users\atala\PycharmProjects\sahi_teknofest_2025\models\rgb_alternative.pt"
        self.rgb_model = YOLO(self.rgb_model_path)
        self.termal_model = YOLO(self.termal_model_path)

        self.rgb_model.to(self.device)
        self.termal_model.to(self.device)
        logging.info('RGB ve Termal modelleri yüklendi')

        # Model seçimi için gerekli değişkenler
        self.current_model = None
        self.model_selected = False
        self.image_type = None
        self.trajectory_estimator = None
        self.processed_images_count = 0
        self.rgb_count = 0
        self.termal_count = 0
        self.max_analysis_images = 10

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

    def color_analysis(self, img):
        """Görüntünün termal mi yoksa normal renkli mi olduğunu analiz eder"""
        if len(img.shape) < 3 or img.shape[2] != 3:
            return "Termal"

        R = img[:, :, 2].astype(float)
        G = img[:, :, 1].astype(float)
        B = img[:, :, 0].astype(float)

        fark_rg = np.abs(R - G)
        fark_rb = np.abs(R - B)
        fark_gb = np.abs(G - B)

        ortalama_fark = (fark_rg.mean() + fark_rb.mean() + fark_gb.mean()) / 3

        logging.info(f"Renk analizi - Ortalama fark: {ortalama_fark}")

        if ortalama_fark <= 2.0:
            return "Termal"
        else:
            return "Normal"

    def detect_img_type(self, image_path):
        """Görüntü tipini tespit eder"""
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Görüntü açılamadı: {image_path}")
            return "Bilinmeyen"

        sonuc = self.color_analysis(img)
        logging.info(f"Görüntü tipi: {image_path} -> {sonuc}")
        return sonuc

    def analyze_first_images_and_select_model(self, image_path):
        """İlk 10 görüntüyü analiz ederek model seçimi yapar"""
        if self.processed_images_count < self.max_analysis_images:
            goruntu_tipi = self.detect_img_type(image_path)

            if goruntu_tipi == "Termal":
                self.termal_count += 1
                # Termal görüntü tespit edilirse hemen termal model kullan
                current_model_for_this_image = self.termal_model
                logging.info(f"Termal görüntü tespit edildi, termal model kullanılıyor: {image_path}")
            elif goruntu_tipi == "Normal":
                self.rgb_count += 1
                # Normal görüntü tespit edilirse RGB model kullan
                current_model_for_this_image = self.rgb_model
                logging.info(f"Normal görüntü tespit edildi, RGB model kullanılıyor: {image_path}")
            else:
                # Bilinmeyen durumda varsayılan olarak RGB model kullan
                current_model_for_this_image = self.rgb_model
                logging.warning(f"Bilinmeyen görüntü tipi, varsayılan RGB model kullanılıyor: {image_path}")

            self.processed_images_count += 1
            logging.info(
                f"Analiz edilen görüntü: {self.processed_images_count}/{self.max_analysis_images}, RGB: {self.rgb_count}, Termal: {self.termal_count}")

            # 10 görüntü analiz edildiğinde model seçimini kesinleştir
            if self.processed_images_count >= self.max_analysis_images:
                self.finalize_model_selection()

            return current_model_for_this_image

        # Eğer 10 görüntü analizi tamamlandıysa seçilmiş modeli döndür
        return self.current_model

    def finalize_model_selection(self):
        """İlk 10 görüntünün analizi sonucunda model seçimini kesinleştir"""
        if self.rgb_count > self.termal_count:
            self.current_model = self.rgb_model
            self.image_type = "Normal"
            model_name = os.path.basename(self.rgb_model_path)
            logging.info(f"RGB model seçildi - {model_name} (RGB: {self.rgb_count}, Termal: {self.termal_count})")

            # RGB Visual Odometry import et
            try:
                from .rgb_visual_odometry import DroneTrajectoryEstimator
                self.trajectory_estimator = DroneTrajectoryEstimator()
                logging.info("RGB Visual Odometry yüklendi")
            except ImportError as e:
                logging.error(f"RGB Visual Odometry yüklenemedi: {e}")

        else:
            self.current_model = self.termal_model
            self.image_type = "Termal"
            model_name = os.path.basename(self.termal_model_path)
            logging.info(f"Termal model seçildi - {model_name} (RGB: {self.rgb_count}, Termal: {self.termal_count})")

            # Termal Visual Odometry import et
            try:
                from .termal_visual_odometry import DroneTrajectoryEstimator
                self.trajectory_estimator = DroneTrajectoryEstimator()
                logging.info("Termal Visual Odometry yüklendi")
            except ImportError as e:
                logging.error(f"Termal Visual Odometry yüklenemedi: {e}")

        self.model_selected = True

    def boxes_overlap(self, box1, box2):
        """İki kutunun çakışıp çakışmadığını kontrol eder (landingg.py'den alınan fonksiyon)"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        return x2_min > x1_max and y2_min > y1_max

    def check_bounding_box_overlap_only(self, detections, current_idx):
        """Termal model için sadece bounding box çakışması kontrolü"""
        # UAP/UAI alanlarını ve diğer nesneleri ayır
        landing_zones = [det for det in detections if det[0] in ["uap", "uai"]]
        other_objects = [det for det in detections if det[0] not in ["uap", "uai"]]

        # Eğer hiç UAP/UAI yoksa iniş yapılamaz
        if not landing_zones:
            return "0"

        # İlk UAP/UAI alanını kontrol et
        zone_bbox = landing_zones[0][1]

        # Diğer nesnelerle çakışma kontrolü
        overlap = any(self.boxes_overlap(zone_bbox, obj[1]) for obj in other_objects)

        # Çakışma yoksa iniş yapılabilir
        return "0" if overlap else "1"

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files, frame):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Ornek)
        if prediction.image_url.startswith("http"):
            image_url = prediction.image_url
        else:
            image_url = evaluation_server_url.rstrip("/") + "/media/" + prediction.image_url.lstrip("/")

        # Görüntüyü indir
        self.download_image(image_url, images_folder, images_files)
        frame = frame
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti ve pozisyon kestirim modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction, health_status, frame)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction, health_status, frame):
        session_name = self.connectionHandler.get_session_name()
        image_filename = prediction.image_url.split('/')[-1]
        image_path = os.path.join('_images', session_name, image_filename)

        # Model seçimi - ilk 10 görüntüyü analiz et
        if not self.model_selected:
            selected_model = self.analyze_first_images_and_select_model(image_path)
            if selected_model is None:
                # Henüz model seçimi yapılmadı, geçici olarak RGB model kullan
                selected_model = self.rgb_model
        else:
            # Model zaten seçili, direkt kullan
            selected_model = self.current_model

        # Seçilen model ile nesne tespiti
        results = selected_model.predict(source=image_path, save=False, verbose=False, device="cuda")
        frame = frame
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return prediction  # hiç tespit yoksa

        boxes = results[0].boxes.xyxy.cpu().numpy()  # numpy format
        classes = results[0].boxes.cls.cpu().numpy()
        detections = []
        for idx in range(len(boxes)):
            cls_id = int(classes[idx])
            class_name = selected_model.names[cls_id]
            objects = [class_name, boxes[idx]]
            detections.append(objects)
        for idx in range(len(boxes)):
            class_id = classes[idx]
            class_name = selected_model.names[class_id]  # class_name artık her nesne için alınır

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

            pred_translation_x = 0.0
            pred_translation_y = 0.0
            pred_translation_z = 0.0

        if str(health_status) == '0' and self.trajectory_estimator is not None:
            frame_bgr = cv2.imread(image_path)
            print(f"Image path: {image_path}, image loaded: {frame_bgr is not None}")
            if frame_bgr is not None:
                estimated_position = self.trajectory_estimator.process_frame(frame_bgr, 0, None)
                print(f"Estimated position from process_frame: {estimated_position}")
                if estimated_position is not None and not np.any(np.isnan(estimated_position)):
                    pred_translation_x = float(estimated_position[0])
                    pred_translation_y = float(estimated_position[1])
                    pred_translation_z = float(estimated_position[2])
                else:
                    print("[UYARI] Estimated position is None or contains NaN, setting zeros.")
                    pred_translation_x = pred_translation_y = pred_translation_z = 0.0
            else:
                print("[HATA] Image could not be loaded, setting zeros.")
                pred_translation_x = pred_translation_y = pred_translation_z = 0.0
        else:
            pred_translation_x = prediction.gt_translation_x
            pred_translation_y = prediction.gt_translation_y
            pred_translation_z = prediction.gt_translation_z

        # Translation icin hesaplanilan degerleri sunucuya gondermek icin ilgili objeye dolduralim.
        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y, pred_translation_z)
        prediction.add_translation_object(trans_obj)

        return prediction