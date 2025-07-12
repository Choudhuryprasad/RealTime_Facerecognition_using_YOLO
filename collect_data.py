import cv2
import urllib.request
import numpy as np
import os

classifier = cv2.CascadeClassifier(r"C:/Users/khadi/Desktop/RTFRUYOLO_V8/haarcascade_frontalface_default.xml")

url = "http://192.0.0.4:8080/shot.jpg" 


data = []

if not os.path.exists("images"):
    os.makedirs("images")

while len(data) < 100:
    try:
        image_from_url = urllib.request.urlopen(url)
        frame = np.array(bytearray(image_from_url.read()), np.uint8)
        frame = cv2.imdecode(frame, -1)
        
        face_points = classifier.detectMultiScale(frame, 1.3, 5)
        
        if len(face_points) > 0:
            for (x, y, w, h) in face_points:
                face_frame = frame[y:y+h, x:x+w]
                cv2.imshow("Only face", face_frame)
                if len(data) < 100:
                    print(len(data) + 1, "/ 100")
                    data.append(face_frame)
                    break
                
        cv2.putText(frame, str(len(data)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255))
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(30) == ord("q"):
            break
            
    except Exception as e:
        print("Error:", e)
        break

cv2.destroyAllWindows()

if len(data) == 100:
    name = input("Enter Face holder name: ")
    for i, face in enumerate(data):
        cv2.imwrite(f"images/{name}_{i}.jpg", face)
    print("Data collection complete.")
else:
    print("Insufficient data collected.")