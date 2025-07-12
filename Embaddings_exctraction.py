import os
import numpy as np
import json
import cv2
from keras_facenet import FaceNet


embedder = FaceNet()

def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

def save_person_embeddings(person_name, folder_path, output_file='known_embeddings.json'):
    embeddings = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = preprocess_img(img)
                embedding = embedder.embeddings([img])
                embeddings.append(embedding[0].tolist())

    person_entry = {'name': person_name, 'embeddings': embeddings}

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            known_data = json.load(f)
    else:
        known_data = []

    known_data.append(person_entry)
    
    with open(output_file, 'w') as f:
        json.dump(known_data, f, indent=4)

person_name = "Sir"
person_folder = "C:/Users/khadi/Desktop/RTFRUYOLO_V8/img_dataset/Sir"
save_person_embeddings(person_name, person_folder)