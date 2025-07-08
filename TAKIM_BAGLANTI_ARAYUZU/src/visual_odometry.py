import cv2
import numpy as np


class CameraParams:
    """Kamera parametrelerini tutan sınıf"""
    INTRINSIC_MATRIX = np.array([
        [2792.2, 0, 1988.0],
        [0, 2795.2, 1562.2],
        [0, 0, 1]
    ], dtype=np.float64)

    DIST_COEFFS = np.array([0.0798, -0.1867, 0, 0, 0], dtype=np.float64)
    IMAGE_SIZE = (1920, 1080)  # width, height


def undistort_image(image):
    """Görüntüyü undistort eden fonksiyon"""
    return cv2.undistort(image, CameraParams.INTRINSIC_MATRIX, CameraParams.DIST_COEFFS)


class DroneTrajectoryEstimator:
    """Drone yörünge tahmin sınıfı - Yarışma versiyonu"""

    # Parametreler
    MATCH_DISTANCE_THRESHOLD = 0.7  # Daha katı eşleşme
    RANSAC_REPROJ_THRESHOLD = 2.0  # Daha düşük eşik
    MIN_MATCH_COUNT = 20  # Minimum eşleşme sayısı

    def __init__(self, manual_scale=0.033164 * (116.350579 / 204.89290647630384)):
        # ORB detector - daha iyi parametreler
        self.orb = cv2.ORB_create(
            nfeatures=2000,  # Daha fazla feature
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )

        # BFMatcher kullan (FLANN yerine daha stabil)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Durum değişkenleri
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None

        # Pozisyon tracking (XYZ - Z ekseni için)
        self.current_position = np.zeros(3, dtype=np.float64)  # [x, y, z]

        # Manuel scale ayarları
        self.scale = np.array([manual_scale, manual_scale])
        self.is_calibrated = True

        # Filtreleme için
        self.velocity_filter = []
        self.max_velocity = 50.0  # pixel/frame maksimum hız

        # Z ekseni için son alınan değer
        self.last_known_z = 0.0

    def process_frame(self, frame, health_status, gt_position=None):
        """
        Ana frame işleme fonksiyonu

        Args:
            frame: Gelen frame
            health_status: Sağlık durumu (1: GPS var, 0: GPS yok)
            gt_position: GPS pozisyonu (health_status=1 iken xyz koordinatları)

        Returns:
            np.array: Mevcut pozisyon [x, y, z]
        """

        # Sağlık durumu 1 ise (GPS var) ground truth kullan
        if health_status == 1 and gt_position is not None:
            self.current_position = gt_position.copy()
            # Z değerini kaydet (sonraki GPS olmayan frame'ler için)
            self.last_known_z = gt_position[2]
            return self.current_position

        # Sağlık durumu 0 ise (GPS yok) visual odometry kullan
        # Gri tonlamaya çevir
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Görüntüyü undistort et
        undistorted = undistort_image(gray)

        # Keypoint ve descriptor hesapla
        keypoints, descriptors = self.orb.detectAndCompute(undistorted, None)

        if descriptors is None or len(keypoints) < 10:
            print("Yeterli keypoint bulunamadı!")
            # Z değerini koru
            self.current_position[2] = self.last_known_z
            return self.current_position

        # İlk frame ise
        if self.prev_descriptors is None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_frame = undistorted
            # Z değerini koru
            self.current_position[2] = self.last_known_z
            return self.current_position

        # Feature matching
        matches = self.match_features(self.prev_descriptors, descriptors)

        if len(matches) < self.MIN_MATCH_COUNT:
            print(f"Yeterli eşleşme bulunamadı: {len(matches)}")
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_frame = undistorted
            # Z değerini koru
            self.current_position[2] = self.last_known_z
            return self.current_position

        # İyi eşleşmeleri filtrele
        good_matches = self.filter_good_matches(matches)

        if len(good_matches) < self.MIN_MATCH_COUNT:
            print(f"Yeterli iyi eşleşme bulunamadı: {len(good_matches)}")
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_frame = undistorted
            # Z değerini koru
            self.current_position[2] = self.last_known_z
            return self.current_position

        # Eşleşen noktaları al
        src_pts, dst_pts = self.get_matched_points(good_matches, keypoints)

        # Affine transformation hesapla
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.RANSAC_REPROJ_THRESHOLD,
            confidence=0.99
        )

        if M is not None and inliers is not None:
            # Translation vektörünü al
            dx, dy = M[0, 2], M[1, 2]
            dx = -dx  # X ekseni düzeltmesi

            # Outlier kontrolü
            velocity_magnitude = np.sqrt(dx**2 + dy**2)
            if velocity_magnitude > self.max_velocity:
                print(f"Çok yüksek hız tespit edildi: {velocity_magnitude:.2f}")
                dx *= (self.max_velocity / velocity_magnitude)
                dy *= (self.max_velocity / velocity_magnitude)

            # Scale uygula (sadece X ve Y için)
            scaled_dx = dx * self.scale[0]
            scaled_dy = dy * self.scale[1]

            # Pozisyonu güncelle (sadece X ve Y)
            self.current_position[0] += scaled_dx
            self.current_position[1] += scaled_dy

        # Z değerini koru
        self.current_position[2] = self.last_known_z

        # Bir sonraki frame için güncelle
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_frame = undistorted

        return self.current_position

    def match_features(self, desc1, desc2):
        """Feature matching fonksiyonu"""
        if desc1 is None or desc2 is None:
            return []

        try:
            # BFMatcher kullan
            matches = self.matcher.match(desc1, desc2)

            # Mesafeye göre sırala
            matches = sorted(matches, key=lambda x: x.distance)

            return matches
        except Exception as e:
            print(f"Matching hatası: {e}")
            return []

    def filter_good_matches(self, matches):
        """İyi eşleşmeleri filtrele"""
        if len(matches) < 2:
            return matches

        # İstatistiksel filtreleme
        distances = [m.distance for m in matches]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Outlier'ları kaldır
        threshold = mean_dist + 0.5 * std_dist
        good_matches = [m for m in matches if m.distance < threshold]

        # En iyi %70'ini al
        num_good = int(len(good_matches) * 0.7)
        return good_matches[:num_good]

    def get_matched_points(self, matches, current_keypoints):
        """Eşleşen noktaları al"""
        src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in matches])

        return src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2)

    def get_current_position(self):
        """Mevcut pozisyonu döndür"""
        return self.current_position.copy()

    def reset(self):
        """Estimator'ı sıfırla"""
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        self.current_position = np.zeros(3, dtype=np.float64)
        self.last_known_z = 0.0


# Yarışma için kullanım örneği
def competition_example():
    """Yarışma senaryosu için örnek kullanım"""

    # Estimator oluştur
    estimator = DroneTrajectoryEstimator()

    # Örnek frame işleme döngüsü
    for frame_idx in range(2250):  # Toplam 2250 frame (5 dakika)

        # Frame'i al (gerçek uygulamada server'dan gelecek)
        # frame = get_frame_from_server()  # Bu sizin implement edeceğiniz fonksiyon

        # Sağlık durumu ve ground truth al (server'dan gelecek)
        # health_status, gt_position = get_health_and_gt_from_server()

        # Örnek değerler (gerçek uygulamada server'dan gelecek)
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)  # Örnek frame

        # Dinamik GPS durumu - sadece health_status'a göre karar
        # İlk kısımda daha fazla GPS var, sonra azalır (örnek pattern)
        if frame_idx < 600:  # İlk kısım - daha fazla GPS
            health_status = 1 if frame_idx % 3 < 2 else 0  # %67 GPS var
        else:  # Son kısım - daha az GPS
            health_status = 1 if frame_idx % 10 < 2 else 0  # %20 GPS var

        # GPS varsa pozisyon ver
        gt_position = np.array([frame_idx * 0.1, frame_idx * 0.05, 50.0]) if health_status == 1 else None

        # Frame'i işle
        estimated_position = estimator.process_frame(frame, health_status, gt_position)

        # Pozisyonu server'a gönder
        # send_position_to_server(estimated_position)

        # Debug için yazdır
        if frame_idx % 100 == 0:
            status_text = "GPS VAR" if health_status == 1 else "GPS YOK"
            print(
                f"Frame {frame_idx}: {status_text}, Position=({estimated_position[0]:.2f}, {estimated_position[1]:.2f}, {estimated_position[2]:.2f})")

    print("Yarışma simülasyonu tamamlandı!")


if __name__ == "_main_":
    competition_example()