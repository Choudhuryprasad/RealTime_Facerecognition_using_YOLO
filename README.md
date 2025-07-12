
# YOLOv8 Face Recognition System

A face recognition system that uses YOLOv8 for face detection and facial embeddings for recognition. This project includes tools for data collection, embedding generation, and real-time recognition via webcam or video.

## Features

- **Face Detection** using YOLOv8 (`yolov8m-face.pt`)
- **Face Recognition** with facial embeddings
- Real-time face capture and recognition
- Easy data collection and embedding generation
- Jupyter notebook for experimentation

## Directory Structure

```
RTFRUYOLO_V8/
├── app.py                         # Main face recognition application
├── collect_data.py               # Collect face images from webcam
├── Embaddings_exctraction.py     # Generate embeddings from collected data
├── test.py                       # Script for testing recognition
├── TheRealDeal.ipynb             # Notebook for testing and visualization
├── yolov8m-face.pt               # Pretrained YOLOv8 model
├── known_embeddings.json         # Stored facial embeddings
├── haarcascade_frontalface_default.xml # Haar cascade for fallback face detection
├── img_dataset/                  # Collected face images (training data)
│   └── <PersonName>/<image>.jpg
├── DOC-20241111-WA0006.pdf       # Project documentation (PDF)
├── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip
- A working webcam (for real-time capture)

### Install Dependencies

```bash
pip install -r requirements.txt
```

_If `requirements.txt` is not available, install manually:_

```bash
pip install ultralytics opencv-python numpy scikit-learn imutils
```

## Usage

### 1. Collect Face Data

```bash
python collect_data.py
```

Follow prompts to capture images for a new identity.

### 2. Generate Embeddings

```bash
python Embaddings_exctraction.py
```

Creates `known_embeddings.json` from the collected face data.

### 3. Run Face Recognition

```bash
python app.py
```

Starts the webcam and recognizes known faces in real-time.

### 4. Testing (Optional)

```bash
python test.py
```

Test embeddings or model behavior manually.

## Model

- **YOLOv8 Model**: `yolov8m-face.pt` is used for accurate and fast face detection.

## License

This project is for educational and research use only. Check the YOLOv8 license for model usage.

---

## Credits

- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Face embedding techniques inspired by FaceNet and OpenCV samples
