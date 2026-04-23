import os
from datetime import datetime
import json
from threading import Lock
import cv2 as cv
import numpy as np
import torch
import pygame
import torch.nn as nn
import requests
from torchvision import models
from torchvision import transforms
KNOWN_FACE_WIDTH = 16.0  
FOCAL_LENGTH = 650  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes

# تحميل الأوزان
model.load_state_dict(torch.load(r"D:\deeplearning_course\read_image_write\face_recognation\resnet50.pth", map_location=device))
model = model.to(device)
model.eval()


print(" Model loaded successfully")
def calculate_distance(face_location):
    """
    face_location: (top, right, bottom, left)
    returns distance in CM
    """
    top, right, bottom, left = face_location

    face_width_pixels = right - left

    if face_width_pixels == 0:
        return None

    distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width_pixels
    return round(distance, 2)

alarm_sound_path = r"D:\deeplearning_course\read_image_write\assets_alarm.mp3"
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(alarm_sound_path)
def play_alarm():
    if not pygame.mixer.get_busy():  # يمنع تكرار الصوت كل فريم
        alarm_sound.play()

def stop_alarm():
    alarm_sound.stop()
def preprocess_image (face_input) : 
        face_input = cv.resize(face_input, (224, 224))
        face_input = cv.cvtColor(face_input, cv.COLOR_BGR2RGB)
        face_input = face_input.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        face_input = (face_input - mean) / std
        face_input = np.transpose(face_input, (2, 0, 1))[np.newaxis, :].astype(np.float32)
        return face_input	
    
def predict_tensor(image_tensor):
    image_tensor = torch.from_numpy(image_tensor).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    real_prob  = probs[0][0].item()
    spoof_prob = probs[0][1].item()
    label = 0 if real_prob > spoof_prob else 1

    return label, real_prob, spoof_prob
import os
from datetime import datetime

ATTENDANCE_PATH = r"D:\deeplearning_course\read_image_write\face_recognation\attendace.json"
'''
def mark_attendance(name, file_path=ATTENDANCE_PATH):

    if name.lower() == "unknown":
        return

    now = datetime.now()
    date_today = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")


    # اقرأ الحضور المسجل اليوم
    with open(file_path, "r") as f:
        lines = f.readlines()
        recorded_today = [
            line.split(",")[0]
            for line in lines[1:]
            if date_today in line
        ]

    # سجل الحضور مرة واحدة في اليوم
    if name not in recorded_today:
        with open(file_path, "a") as f:
            f.write(f"{name},{date_today},{time_now}\n")
        print(f"[INFO] Attendance marked for {name}")
        

'''
'''
file_lock = Lock()
def mark_attendance(name, file_path=ATTENDANCE_PATH):

    if name.lower() == "unknown":
        return

    now = datetime.now()
    date_today = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")
    
    with file_lock:
        # لو الملف مش موجود → أنشئه
        if not os.path.exists(file_path):
            data = []
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []

   
        recorded_today = any(
            entry["name"] == name and entry["date"] == date_today
            for entry in data
        )

        # سجل مرة واحدة في اليوم
        if not recorded_today:
            data.append({
                "name": name,
                "date": date_today,
                "time": time_now
            })

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"[INFO] Attendance marked for {name}")
'''
# مسار ملف الحدث (اختياري لتسجيله محليًا)
EVENT_FILE = "attendance.json"

# URL الخاص بالـ Cloudflare Worker
BACKEND_URL = "https://jolly-feather-3cb6.ahmedkoka123476.workers.dev/"

# دالة توليد الحدث
def generate_event(name , distance_cm='', real_pro=''):
    now = datetime.now()
    event = {
        "name": name,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "distance": distance_cm,
        "real_prob": real_pro
    }
    # حفظ الحدث في ملف JSON (اختياري)
    with open(EVENT_FILE, "w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)
    return event

# دالة إرسال الحدث للـ backend
def send_event(event):
    try:
        response = requests.post(BACKEND_URL, json=event, timeout=2)
        if response.status_code == 200:
            print(f"[INFO] Event sent for {event['name']}")
        else:
            print(f"[ERROR] Failed to send event: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] {e}")


def detect_face(frame, faceNet):
    # prepare input blob for the face detector
    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0)) # scale 1 is mean do Normalization  
   
    #عشان نحول الصورة لصيغة يفهمها نموذج الـ DNN
    faceNet.setInput(blob)#دخلنا الصورة للموديل
    detections = faceNet.forward()#)أخدنا كل الـ detections
    

    faces = []
    locs = []


    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # ignore very small / partial detections
            if endX - startX < 50 or endY - startY < 50:
                continue

            # add small padding so partially-out faces keep context
            padX = int(0.1 * (endX - startX))
            padY = int(0.1 * (endY - startY))
            sx = max(0, startX - padX)
            sy = max(0, startY - padY)
            ex = min(w - 1, endX + padX)
            ey = min(h - 1, endY + padY)

            face = frame[sy:ey, sx:ex]
            if face.size == 0:
                continue

            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))


            faces.append(face)
            locs.append((sx, sy, ex, ey))

    # always return locs (empty list if no faces detected)
    return locs