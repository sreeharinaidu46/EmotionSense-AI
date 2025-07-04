# 🎧 EmotionSense AI

EmotionSense AI is a multimodal deep learning application that detects human emotions from both **voice recordings** and their corresponding **text transcriptions** using **TensorFlow**, **Keras**, and **OpenAI Whisper**. The frontend is built using vanilla HTML/CSS/JS for a lightweight UI, while the backend is powered by Flask.

---

## 🚀 Features

- 🎙️ Accepts `.wav` audio files.
- 🧠 Predicts emotion using a trained deep learning model.
- 📄 Displays speech transcription using Whisper.
- 💡 Dynamically changes background color and emoji based on the emotion.
- 🔁 Real-time typing and emoji animation for UX polish.

---

## 🧠 Tech Stack

**Backend:**
- Python
- Flask
- TensorFlow / Keras
- OpenAI Whisper
- Librosa
- Scikit-learn
- Joblib

**Frontend:**
- HTML5
- CSS3
- JavaScript

**Other Tools:**
- Gunicorn
- Pandas, NumPy
- Audio preprocessing via Librosa

---

## 📂 Project Structure

emotion-sense-ai/
├── app.py # Flask app entry point
├── model_loader.py # Loads models and preprocessors
├── templates/
│ └── index.html # Frontend UI
├── static/
│ └── style.css # Frontend styling
├── model/
│ └── multimodal_model/ # Trained Keras model folder
├── data/
│ └── audio_scaler.pkl # Pre-fitted audio scaler (used during inference)
├── requirements.txt # Python dependencies
└── README.md

yaml
Copy
Edit

---

## 🛠️ How to Run Locally

### 1. Clone the Repository


git clone https://github.com/your-username/emotionsense-ai.git
cd emotionsense-ai

2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Requirements

pip install -r requirements.txt

4. Download Whisper Model (if not cached)

import whisper
model = whisper.load_model("base")

5. Run the Flask App

python app.py


Now, open your browser at: http://localhost:5000

🧪 Sample Usage

Upload any .wav file with human speech. The app will:

Transcribe the speech using Whisper

Predict the corresponding emotion

Display it with animation and background change

🙋‍♂️ Author
Sreeharinaidu Rangani
