import cv2
from ultralytics import YOLO
import numpy as np
import math

# --- Kendi Özel YOLOv12 Modelinizin Yüklenmesi ---
# Modelinizin dosya yolunu buraya yazın
custom_model_path = 'teknofest/runs/detect/train4/weights/best.pt'
model = None
try:
    # Ultralytics YOLO ile modeli yükleyin
    model = YOLO(custom_model_path)
    print(f"Model başarıyla yüklendi: {custom_model_path}")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    print("Lütfen model dosyasının yolunu ve formatını kontrol edin.")

if model is None:
    exit()
# --------------------------------------------------

def yuvarlaklik_kontrol_frame(frame):
    """
    Verilen frame'deki 'uai' nesnelerinin yuvarlaklığını kontrol eder ve
    sonuçları konsola yazdırır.

    Args:
        frame (numpy.ndarray): İşlenecek görüntü karesi (BGR formatında).
    """
    yuvarlaklik_esigi_elips = 0.85

    # YOLO ile tespiti gerçekleştirin
    results = model.predict(frame, verbose=False) # verbose=False ile çıktıları gizle

    # Tespit sonuçlarını işleyin
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls.cpu().numpy().item())

            if cls == 12: # 'uai' sınıfının ID'si (modelinize göre değişebilir)
                x_min, y_min, x_max, y_max = map(int, xyxy)
                cropped_image = frame[y_min:y_max, x_min:x_max].copy()

                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                is_yuvarlak = False
                yuvarlaklik_orani = 0.0
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if len(largest_contour) >= 5:
                        ellipse = cv2.fitEllipse(largest_contour)
                        (center, axes, angle) = ellipse
                        major_axis = max(axes)
                        minor_axis = min(axes)
                        if major_axis > 0:
                            yuvarlaklik_orani = minor_axis / major_axis
                            is_yuvarlak = yuvarlaklik_orani >= yuvarlaklik_esigi_elips

                        print(f"'uai' (Güven: {conf:.2f}) - Yuvarlaklık Oranı: {yuvarlaklik_orani:.2f}, Tam Yuvarlak: {is_yuvarlak}")
                    else:
                        print(f"'uai' (Güven: {conf:.2f}) - Yetersiz kontur noktası, yuvarlaklık kontrolü yapılamadı.")
                else:
                    print(f"'uai' (Güven: {conf:.2f}) - Kontur bulunamadı, yuvarlaklık kontrolü yapılamadı.")

                # İsteğe bağlı olarak frame üzerinde görselleştirme yapabilirsiniz
                color = (0, 255, 0) if is_yuvarlak else (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                if contours and len(largest_contour) >= 5:
                    cv2.ellipse(frame, (int(center[0] + x_min), int(center[1] + y_min)), (int(major_axis / 2), int(minor_axis / 2)), angle, 0, 360, color, 2)
    else:
        print("Frame'de hiç nesne tespit edilmedi.")

    # İsteğe bağlı olarak işlenmiş frame'i gösterebilirsiniz
    cv2.imshow("Frame with UAI Detection", frame)
    cv2.waitKey(0) # Ekranda kalması için 0 yapıldı
    cv2.destroyAllWindows() # Pencereyi kapat

# Örnek Kullanım (Tek bir fotoğrafı işlemek için):
if __name__ == "__main__":
    # İşlemek istediğiniz fotoğrafın dosya yolunu buraya yazın
    image_path = "C:/Users/atala/Desktop/test/frame_000052.jpg"
    frame = cv2.imread(image_path)
    if frame is not None:
        yuvarlaklik_kontrol_frame(frame)
    else:
        print(f"Görüntü dosyası yüklenemedi: {image_path}")