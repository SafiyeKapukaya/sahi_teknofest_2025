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
        
        # *** STEP 1: Drone'un yönelim (orientation) durumunu takip etmek için yeni state değişkeni ***
        self.current_angle = 0.0  # Drone'un mutlak yaw açısı (radyan cinsinden)

        # Manuel scale ayarları
        self.scale = np.array([manual_scale, manual_scale])
        self.is_calibrated = True

        # Filtreleme için
        self.velocity_filter = []
        self.max_velocity = 50.0  # pixel/frame maksimum hız

        # Z ekseni için son alınan değer
        self.last_known_z = 0.0
        
        # *** NEW: Z-axis estimation state variables ***
        self.filtered_z = 0.0  # EMA filtered altitude estimate
        self.ema_alpha = 0.1  # EMA filter parameter (0.1 provides good smoothing)

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

        print(f"[DEBUG] process_frame çağrıldı - health_status: {health_status}, gt_position: {gt_position}")
        print(f"[DEBUG] frame shape: {frame.shape if frame is not None else 'None'}")

        # Sağlık durumu 1 ise (GPS var) ground truth kullan
        if health_status == 1 and gt_position is not None:
            print("[DEBUG] GPS modu - GT position kullanılıyor")
            self.current_position = gt_position.copy()
            # Z değerini kaydet (sonraki GPS olmayan frame'ler için)
            self.last_known_z = gt_position[2]
            # *** NEW: Reset EMA filter state with accurate GPS altitude ***
            self.filtered_z = gt_position[2]  # Sync filtered altitude with GPS data
            return self.current_position

        print("[DEBUG] Visual odometry modu başlatılıyor")

        # Sağlık durumu 0 ise (GPS yok) visual odometry kullan
        # Gri tonlamaya çevir
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("[DEBUG] Frame BGR'den Gray'e çevrildi")
        else:
            gray = frame
            print("[DEBUG] Frame zaten grayscale")

        # Görüntüyü undistort et
        undistorted = undistort_image(gray)
        print("[DEBUG] Undistort işlemi tamamlandı")

        # Keypoint ve descriptor hesapla
        keypoints, descriptors = self.orb.detectAndCompute(undistorted, None)
        print(f"[DEBUG] Keypoint sayısı: {len(keypoints) if keypoints else 0}")
        print(f"[DEBUG] Descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")

        if descriptors is None or len(keypoints) < 10:
            print("[DEBUG] Yeterli keypoint bulunamadı - pozisyon korunuyor")
            # Z değerini koru
            self.current_position[2] = self.last_known_z
            print(f"[DEBUG] Dönen pozisyon: {self.current_position}")
            return self.current_position

        # İlk frame ise
        if self.prev_descriptors is None:
            print("[DEBUG] İlk frame - state başlatılıyor")
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_frame = undistorted
            # Z değerini koru
            self.current_position[2] = self.last_known_z
            print(f"[DEBUG] İlk frame pozisyon: {self.current_position}")
            return self.current_position

        # Feature matching
        matches = self.match_features(self.prev_descriptors, descriptors)
        print(f"[DEBUG] Ham match sayısı: {len(matches)}")

        if len(matches) < self.MIN_MATCH_COUNT:
            print(f"[DEBUG] Yeterli eşleşme bulunamadı: {len(matches)} < {self.MIN_MATCH_COUNT}")
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_frame = undistorted
            # Z değerini koru
            self.current_position[2] = self.last_known_z
            print(f"[DEBUG] Dönen pozisyon (yetersiz match): {self.current_position}")
            return self.current_position

        # İyi eşleşmeleri filtrele
        good_matches = self.filter_good_matches(matches)
        print(f"[DEBUG] İyi match sayısı: {len(good_matches)}")

        if len(good_matches) < self.MIN_MATCH_COUNT:
            print(f"[DEBUG] Yeterli iyi eşleşme bulunamadı: {len(good_matches)} < {self.MIN_MATCH_COUNT}")
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_frame = undistorted
            # Z değerini koru
            self.current_position[2] = self.last_known_z
            print(f"[DEBUG] Dönen pozisyon (yetersiz good match): {self.current_position}")
            return self.current_position

        # Eşleşen noktaları al
        src_pts, dst_pts = self.get_matched_points(good_matches, keypoints)
        print(f"[DEBUG] src_pts shape: {src_pts.shape}, dst_pts shape: {dst_pts.shape}")

        # Affine transformation hesapla
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.RANSAC_REPROJ_THRESHOLD,
            confidence=0.99
        )

        print(f"[DEBUG] Affine matrix: {M}")
        print(f"[DEBUG] Inlier sayısı: {np.sum(inliers) if inliers is not None else 0}")

        if M is not None and inliers is not None:
            # Translation vektörünü al
            dx, dy = M[0, 2], M[1, 2]
            dx = -dx  # X ekseni düzeltmesi

            print(f"[DEBUG] Ham translation: dx={dx:.3f}, dy={dy:.3f}")

            # *** STEP 2: Rotasyon bilgisini affine matrix'ten çıkar ***
            # Drone'un frame'ler arası dönüş açısını hesapla
            d_theta = np.arctan2(M[1, 0], M[0, 0])

            # *** NEW: Extract scale factor for Z-axis estimation ***
            # Calculate the scale factor from the affine transformation matrix
            s = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)

            print(f"[DEBUG] Dönüş açısı: {np.degrees(d_theta):.2f}°, Scale: {s:.3f}")

            # Safety check to prevent division by zero or invalid scale values
            if s < 0.01:
                s = 1.0  # Use neutral scale if value is too small

            # Calculate raw altitude estimate based on scale change
            # Scale decrease means drone moved up (objects appear smaller)
            # Scale increase means drone moved down (objects appear larger)
            raw_z = self.current_position[2] / s

            # Apply EMA filter to smooth the noisy raw Z estimate
            self.filtered_z = self.ema_alpha * raw_z + (1 - self.ema_alpha) * self.filtered_z

            # Outlier kontrolü
            velocity_magnitude = np.sqrt(dx ** 2 + dy ** 2)
            if velocity_magnitude > self.max_velocity:
                print(f"[DEBUG] Çok yüksek hız tespit edildi: {velocity_magnitude:.2f}, sınırlandırılıyor")
                dx *= (self.max_velocity / velocity_magnitude)
                dy *= (self.max_velocity / velocity_magnitude)

            # Scale uygula (sadece X ve Y için)
            scaled_dx = dx * self.scale[0]
            scaled_dy = dy * self.scale[1]

            print(f"[DEBUG] Scaled translation: dx={scaled_dx:.6f}, dy={scaled_dy:.6f}")

            # *** STEP 3: Local frame'deki hareketi world frame'e dönüştür ***
            # scaled_dx ve scaled_dy drone'un local (body) frame'indeki hareket
            # Bu değerleri drone'un mevcut yönelimi ile world frame'e çevir
            dx_world = scaled_dx * np.cos(self.current_angle) - scaled_dy * np.sin(self.current_angle)
            dy_world = scaled_dx * np.sin(self.current_angle) + scaled_dy * np.cos(self.current_angle)

            print(f"[DEBUG] World frame translation: dx_world={dx_world:.6f}, dy_world={dy_world:.6f}")

            # *** World frame'deki hareket ile global pozisyonu güncelle ***
            old_position = self.current_position.copy()
            self.current_position[0] += dx_world
            self.current_position[1] += dy_world

            # *** STEP 4: Drone'un mutlak yönelimini güncelle ***
            self.current_angle += d_theta

            # *** NEW: Update Z-axis position with filtered altitude estimate ***
            self.current_position[2] = self.filtered_z

            print(f"[DEBUG] Eski pozisyon: {old_position}")
            print(f"[DEBUG] Yeni pozisyon: {self.current_position}")

            # Debug için yönelim bilgisini yazdır (isteğe bağlı)
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 1

            if self._debug_counter % 50 == 0:  # Her 50 frame'de bir yazdır
                angle_degrees = np.degrees(self.current_angle) % 360
                print(
                    f"Frame {self._debug_counter}: Angle = {angle_degrees:.1f}°, dθ = {np.degrees(d_theta):.2f}°, Scale = {s:.3f}, Z = {self.filtered_z:.2f}")

        else:
            print("[DEBUG] Affine transformation hesaplanamadı")
            # *** NEW: If no valid transformation, keep using filtered Z value ***
            self.current_position[2] = self.filtered_z

        # Bir sonraki frame için güncelle
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_frame = undistorted

        print(f"[DEBUG] Final pozisyon: {self.current_position}")
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

    def get_current_angle(self):
        """Mevcut yönelim açısını döndür (radyan)"""
        return self.current_angle

    def get_current_angle_degrees(self):
        """Mevcut yönelim açısını derece cinsinden döndür"""
        return np.degrees(self.current_angle) % 360

    def reset(self):
        """Estimator'ı sıfırla"""
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        self.current_position = np.zeros(3, dtype=np.float64)
        # *** STEP 5: Reset işleminde yönelim açısını da sıfırla ***
        self.current_angle = 0.0
        self.last_known_z = 0.0
        # *** NEW: Reset filtered altitude state ***
        self.filtered_z = 0.0  # Reset EMA filter state


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

        # GPS varsa pozisyon ver (Z ekseni de değişken altitude simüle eder)
        base_altitude = 50.0 + 10.0 * np.sin(frame_idx * 0.01)  # Varying altitude
        gt_position = np.array([frame_idx * 0.1, frame_idx * 0.05, base_altitude]) if health_status == 1 else None

        # Frame'i işle
        estimated_position = estimator.process_frame(frame, health_status, gt_position)

        # Pozisyonu server'a gönder
        # send_position_to_server(estimated_position)

        # Debug için yazdır
        if frame_idx % 100 == 0:
            status_text = "GPS VAR" if health_status == 1 else "GPS YOK"
            angle_deg = estimator.get_current_angle_degrees()
            print(
                f"Frame {frame_idx}: {status_text}, Position=({estimated_position[0]:.2f}, {estimated_position[1]:.2f}, {estimated_position[2]:.2f}), Angle={angle_deg:.1f}°")

    print("Yarışma simülasyonu tamamlandı!")


if __name__ == "__main__":
    competition_example()