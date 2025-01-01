import cv2
from ultralytics import YOLO

# Load model YOLOv11
  # Ganti dengan path model Anda
model = YOLO('yolov11.pt')

# Inisialisasi webcam
cap = cv2.VideoCapture(0)  # 0 untuk webcam default

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Lakukan deteksi
    results = model(frame)

    # Gambar hasil deteksi
    annotated_frame = results[0].plot()  # Menggambar bounding box dan label

    # Tampilkan frame
    cv2.imshow('Real-time Hand Sign Language Detection', annotated_frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()