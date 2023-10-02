import cv2 as cv
import numpy as np
import os
import pandas as pd
from datetime import datetime

data_folder_path = "D:\\face_detection_attendance\\dataset"
dirs = os.listdir(data_folder_path)

face_regions = []
labels = []

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

label_map = {}
current_id = 0

for dir_name in dirs:
    if dir_name not in label_map:
        label_map[dir_name] = current_id
        current_id += 1

    label_id = label_map[dir_name]
    subject_dir_path = data_folder_path + '\\' + dir_name
    subject_images_names = os.listdir(subject_dir_path)

    for image_name in subject_images_names:
        image_path = subject_dir_path + "/" + image_name
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        detected_faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(detected_faces) == 0:
            continue
        (x, y, w, h) = detected_faces[0]
        face_region = image[y:y+h, x:x+w]
        face_regions.append(face_region)
        labels.append(label_id)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_regions, np.array(labels))

def save_to_excel(name):
    filename = "attendance_record.xlsx"
    if os.path.exists(filename):
        df = pd.read_excel(filename)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    if name not in df['Name'].values:
        current_time = datetime.now()
        date_str = current_time.strftime("%Y-%m-%d")
        time_str = current_time.strftime("%H:%M:%S")
        new_record = {"Name": name, "Date": date_str, "Time": time_str}
        df = df.append(new_record, ignore_index=True)
        df.to_excel(filename, index=False)

cap = cv.VideoCapture(0)
recorded_faces = set()

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        id, confidence = face_recognizer.predict(face_roi)
        if confidence < 100:  # threshold for recognition
            predicted_name = [name for name, label in label_map.items() if label == id][0]
            if predicted_name not in recorded_faces:
                save_to_excel(predicted_name)
                recorded_faces.add(predicted_name)
        else:
            predicted_name = "New Face"
        cv.putText(frame, predicted_name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow("Face Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
