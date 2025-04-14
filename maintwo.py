from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO("models/bu.pt")

# Video input
input_video = "C:/Users/safiy/PycharmProjects/teknofest_uyz2025/test_videos/5741005-uhd_3840_2160_25fps.mp4"
cap = cv2.VideoCapture(input_video)

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize dimensions
resize_width = 600
resize_height = 500

# Camera intrinsic matrix (adjust these based on real camera params)
K = np.array([[600, 0, resize_width / 2],
              [0, 600, resize_height / 2],
              [0, 0, 1]], dtype=np.float32)

# Initialize variables
camera_trajectory = [np.eye(4)]  # Store pose transformation matrices
prev_gray = None
prev_points = None

# Lucas-Kanade Optical Flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

plt.figure()
plt.title("2D Map")
plt.xlabel("X")
plt.ylabel("Y")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert to grayscale
    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # YOLO object detection
    results = model(resized_frame)
    annotated_frame = results[0].plot()

    # Initialize feature points for Optical Flow
    if prev_gray is not None:
        # Compute optical flow (track feature points)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

        # Select good points
        good_prev = prev_points[status.flatten() == 1]
        good_next = next_points[status.flatten() == 1]

        if len(good_prev) > 8:
            # Compute Essential Matrix with better RANSAC filtering
            E, mask = cv2.findEssentialMat(good_prev, good_next, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            # Recover camera pose
            _, R, t, mask = cv2.recoverPose(E, good_prev, good_next, K)

            # Normalize translation (assume constant scale)
            scale = np.linalg.norm(t)
            if scale < 0.1:  # Ignore small movements
                scale = 1.0
            t = t / scale  # Normalize translation

            # Update global transformation
            last_transform = camera_trajectory[-1]
            new_transform = np.eye(4)
            new_transform[:3, :3] = R
            new_transform[:3, 3] = t.flatten()

            camera_trajectory.append(last_transform @ new_transform)  # Accumulate motion

    # Update previous frame and points
    prev_gray = gray
    prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)

    # Display YOLO detections
    cv2.imshow("YOLO Detection", annotated_frame)

    # Plot 2D trajectory
    if len(camera_trajectory) > 1:
        x_vals = [pose[0, 3] for pose in camera_trajectory]
        z_vals = [pose[2, 3] for pose in camera_trajectory]
        plt.plot(x_vals, z_vals, 'b-')
        plt.draw()
        plt.pause(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.show()
