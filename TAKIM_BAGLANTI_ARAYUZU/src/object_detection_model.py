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
        self.model = YOLO(r"C:\Users\safiy\havacilikta-yapay-zeka-yarismasi\models\mehmet.pt") # Örnektir!

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

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files,frame):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Ornek)
        self.download_image(evaluation_server_url + "media" + prediction.image_url, images_folder, images_files)
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        frame=frame
        # Nesne tespiti ve pozisyon kestirim modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction, health_status,frame)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction, health_status,frame):
        # Modelinizle bu fonksiyon içerisinde tahmin yapınız.
        results = self.model.predict()
        frame=frame
        # Burada örnek olması amacıyla 2 adet tahmin yapıldığı simüle edilmiştir.
        # Yarışma esnasında modelin tahmin olarak ürettiği sonuçlar kullanılmalıdır.

        # for i in results: # gibi
        for i in results:
            uap_or_uai_exists = False
            if i[0] in  ['gemi', 'ikiteker', 'ismakinesi', 'kamyon', 'otobus', 'otomobil', 'scooter', 'tekne', 'vagon']:
                i[0]=0
            elif i[0] == "insan":
                i[0]=1
            elif i[0]=="uai":
                i[0]=3
                uap_or_uai_exists = True
            else:
                i[0]=2
                uap_or_uai_exists = True

            cls=i[0]
            if cls not in [2,3]:
                landing_status = "-1"
            else:
                landing_status = check_landing_conditions(i,frame)

               # Tahmin edilen nesnenin sınıfı classes sözlüğü kullanılarak atanmalıdır.
         # Tahmin edilen nesnenin inilebilir durumu landing_statuses sözlüğü kullanılarak atanmalıdır.
            top_left_x = i[2][0]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            top_left_y = i[2][1]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            bottom_right_x = i[2][2]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.
            bottom_right_y = i[2][3]  # Örnek olması için rastgele değer atanmıştır. Modelin sonuçları kullanılmalıdır.

            # Modelin tespit ettiği herbir nesne için bir DetectedObject sınıfına ait nesne oluşturularak
            # tahmin modelinin sonuçları parametre olarak verilmelidir.
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y
                                   )

            # Modelin tahmin ettiği her nesne prediction nesnesi içerisinde bulunan detected_objects listesine eklenmelidir.
            prediction.add_detected_object(d_obj)

        # Health Status biti hava aracinin uydu haberlesmesinin saglikli olup olmadigini gosterir.
        # Health Status 0 ise sistem calismali 1 ise gelen verinin aynisini gondermelidir.

        if health_status == '0':
            # Takimlar buraya kendi gelistirdikleri algoritmalarin sonuclarini entegre edebilirler.
            pred_translation_x = random.randint(1,
                                                10)  # Ornek olmasi icin rastgele degerler atanmistir takimlar kendi sonuclarini kullanmalidirlar.
            pred_translation_y = random.randint(1,
                                                10)  # Ornek olmasi icin rastgele degerler atanmistir takimlar kendi sonuclarini kullanmalidirlar.
        else:
            pred_translation_x = prediction.gt_translation_x
            pred_translation_y = prediction.gt_translation_y


        # Translation icin hesaplanilan degerleri sunucuya gondermek icin ilgili objeye dolduralim.
        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y)
        prediction.add_translation_object(trans_obj)

        return prediction