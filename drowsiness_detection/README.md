# 💤 Drowsiness Detection using Eye State and Webcam

This project detects driver drowsiness based on eye state (open or closed) using real-time webcam input and deep learning.

---

## 📦 Features

- Detects if both eyes are closed using facial landmarks.
- Triggers a warning message and plays an alert sound.
- Supports toggling sound on/off using keyboard.
- Real-time video display with bounding box color feedback.

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/khangnguyen1521/FaceDetection.git
cd FaceDetection

---
### 2️⃣ Create virtual environment (optional but recommended)
python -m venv .venv
---

---
33️⃣ Activate the virtual environment
On Windows:
.venv\Scripts\activate

###
On macOS/Linux:
source .venv/bin/activate
---

---
4️⃣ Install dependencies
pip install -r requirements.txt
---

📁 Folder Structure
bash
Copy
Edit
FaceDetection/
├── drowsiness_detection/
│   ├── alert.mp3                         # Sound played on drowsiness
│   ├── shape_predictor_68_face_landmarks.dat  # Dlib facial landmark model
│   ├── weights.149-0.01.hdf5            # Trained Keras model for eye state
│   ├── sleep_detect.py                  # Main detection script
│   └── setup.txt                        # Optional notes/dependencies
├── .gitignore
├── requirements.txt
└── README.md
🎮 Controls
Q: Quit the app

M: Toggle alert sound ON/OFF



