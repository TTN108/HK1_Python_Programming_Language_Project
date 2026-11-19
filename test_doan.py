import cv2
import torch
import ultralytics
import pyttsx3

print("--- KIỂM TRA MÔI TRƯỜNG ĐỒ ÁN THỊ GIÁC MÁY TÍNH ---")
print("1. OpenCV version     :", cv2.__version__)
print("2. PyTorch version    :", torch.__version__)
print("3. Ultralytics (YOLO) :", ultralytics.__version__)

# Kiểm tra giọng đọc
print("4. Kiểm tra loa phát âm thanh...")
engine = pyttsx3.init()
engine.say("Môi trường nhận diện mệnh giá tiền đã sẵn sàng!")
engine.runAndWait()

print("Chúc mừng bạn! Môi trường học tập và làm đồ án đã hoàn tất 100%!")