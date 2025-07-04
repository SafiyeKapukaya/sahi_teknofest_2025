from ultralytics import YOLO
import cv2
import numpy as np

# Bu satırı kendi projenize göre düzenlemeniz gerekebilir.
from algorithms.landing import check_landing_conditions

# --- Örnek check_landing_conditions Fonksiyonu ---
# Kodun çalışabilmesi için geçici bir fonksiyon ekledim.
# Kendi orijinal fonksiyonunuzu kullanmaya devam edebilirsiniz.


# Modeli yükle
model = YOLO("models/xlbest.pt")


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


# Video giriş/çıkış ayarları
# Lütfen video yolunu kendi sisteminize göre güncelleyin.
# input_video = r"C:\path\to\your\video.mp4"
# output_video = r"C:\path\to\your\output.mp4"
# Örnek yollar, kendi yollarınızla değiştirmeyi unutmayın.
input_video = r"test_videos\video.MP4"  # Kendi video yolunuzu buraya yazın
output_video = "output.mp4"  # Çıktı video yolunu buraya yazın

# Video yolunu kontrol et
try:
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Hata: Video dosyası açılamadı veya bulunamadı: {input_video}")
except Exception as e:
    print(e)
    # Örnek bir video dosyası oluşturma (eğer dosya yoksa)
    print("Örnek bir video dosyası (siyah ekran) oluşturuluyor: test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_temp = cv2.VideoWriter('test_video.mp4', fourcc, 30, (640, 480))
    for _ in range(90):  # 3 saniyelik video
        out_temp.write(np.zeros((480, 640, 3), dtype=np.uint8))
    out_temp.release()
    cap = cv2.VideoCapture('test_video.mp4')

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

resize_width = 960
resize_height = 720

# Video döngüsü
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- DEĞİŞİKLİK BURADA ---
    # Modeli GPU üzerinde çalıştırmak için device='cuda' veya device=0 parametresini ekliyoruz.
    # Bilgisayarınızda CUDA destekli bir NVIDIA GPU ve gerekli sürücülerin/kütüphanelerin
    # (PyTorch CUDA versiyonu) kurulu olması gerekir.
    results = model(frame, device='cuda')[0]

    # Sonuçları CPU'ya alıp numpy array'e çeviriyoruz. Bu satır doğru.
    boxes = results.boxes.data.cpu().numpy()

    selected_boxes = []
    if boxes.shape[0] > 0:  # Eğer herhangi bir kutu tespit edildiyse devam et
        used = np.zeros(len(boxes), dtype=bool)

        # Aynı nesnenin birden fazla kutusunu eleyerek en güvenilirini al (Non-Maximum Suppression benzeri mantık)
        # Not: Ultralytics'in kendi NMS'i zaten var, bu özel bir filtreleme adımıdır.
        sorted_indices = np.argsort(boxes[:, 4])[::-1]  # Güven skoruna göre sırala
        boxes = boxes[sorted_indices]

        for i in range(len(boxes)):
            if used[i]:
                continue

            selected_boxes.append(boxes[i])
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if not used[j]:
                    iou = calculate_iou(boxes[i][:4], boxes[j][:4])
                    if iou > 0.5:  # Yüksek IOU'ya sahip kutuları ele
                        used[j] = True

    # Algılanan kutular ve etiketler
    detections = []
    for box in selected_boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls = int(cls)
        label = model.names[cls]

        detections.append({'label': label, 'bbox': (x1, y1, x2, y2)})

        # Renk seçimi: farklı sınıflara göre
        color_map = {
            "insan": (255, 255, 0),  # Sarı
            "uap": (0, 0, 255),  # Kırmızı
            "uai": (255, 0, 255),  # Mor
            "kamyon": (255, 0, 0),  # Mavi
        }
        color = color_map.get(label, (0, 255, 255))  # Varsayılan: Camgöbeği

        # Kutu çizimi ve etiket yazımı
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Kalınlık artırıldı
        label_text = f"{label} {conf:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Font boyutu ayarlandı

    # İniş kontrolü yazılarını çiz
    statuses = check_landing_conditions(detections)
    for (x1, y1, x2, y2), msg in [(s[0], s[1]) for s in statuses]:
        color = (0, 0, 255) if "UYGUN DEGIL" in msg else (0, 255, 0)
        cv2.putText(frame, msg, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Görüntüyü yeniden boyutlandır ve göster
    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    out.write(frame)
    cv2.imshow("Object Detection (GPU)", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kapatma işlemleri
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"İşlem tamamlandı. Video '{output_video}' olarak kaydedildi.")