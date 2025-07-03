# model_loader.py
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained multimodal model
MODEL_PATH = "model/multimodal_model"
model = tf.keras.models.load_model(MODEL_PATH)

# Load fitted scalers and vectorizers
audio_scaler = joblib.load("data/audio_scaler.pkl")
text_vectorizer = joblib.load("data/text_vectorizer.pkl")
label_encoder = joblib.load("data/label_encoder.pkl")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def predict_emotion(audio_path, transcript):
    audio_features = extract_features(audio_path).reshape(1, -1)
    audio_features = audio_scaler.transform(audio_features)

    text_features = text_vectorizer.transform([transcript]).toarray()

    prediction = model.predict([audio_features, text_features])
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]