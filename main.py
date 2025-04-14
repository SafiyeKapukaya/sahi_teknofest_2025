from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2

#model yüklemesi
model = YOLO("models/bu.pt")
# test verisine referans
"""results = model("C:/Users/safiy/PycharmProjects/teknofest_uyz2025/dataset/test/images/30020_jpg.rf.a0247541288a0c0cf83f455d189048c5.jpg")
annotated_image = results[0].plot()
plt.imshow(annotated_image)
plt.axis("off")
plt.show()"""

#video üzerinden test
input_video = "C:/Users/safiy/PycharmProjects/teknofest_uyz2025/test_videos/5741005-uhd_3840_2160_25fps.mp4"
output_video = "C:/Users/safiy/PycharmProjects/teknofest_uyz2025/test_videos/ouputs"
cap = cv2.VideoCapture(input_video)

#video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

resize_width = 600
resize_height = 500

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    resized_frame = cv2.resize(annotated_frame, (resize_width, resize_height))
    out.write(annotated_frame)
    cv2.imshow("YOLO Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()



