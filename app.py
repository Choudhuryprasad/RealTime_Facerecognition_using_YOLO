import streamlit as st
import cv2
import os
import json
import numpy as np
import torch
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

face_detector = YOLO('yolov8m-face.pt').to(device)

embedder = FaceNet()

def load_known_embeddings(json_file='known_embeddings.json'):
    with open(json_file, 'r') as f:
        return json.load(f)

def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

def recognize_face(embedding, known_data, threshold=0.5):
    highest_similarity = 0
    recognized_name = "Unknown"
    
    for person in known_data:
        name = person['name']
        for known_embedding in person['embeddings']:
            sim = cosine_similarity([embedding], [known_embedding])[0][0]
            if sim > highest_similarity and sim > threshold:
                highest_similarity = sim
                recognized_name = name

    return recognized_name

known_embeddings_data = load_known_embeddings()

st.title("Real-Time Face Detection and Recognition")

cap = cv2.VideoCapture(0)
stframe = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture frame from webcam.")
        break

    results = face_detector(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            
            if conf > 0.5:
                face = frame[y1:y2, x1:x2]
                face_resized = cv2.resize(preprocess_img(face), (160, 160))
                embedding = embedder.embeddings([face_resized])[0]
                recognized_name = recognize_face(embedding, known_embeddings_data)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, recognized_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()