import os
import numpy as np
import librosa
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# --- File paths ---
DATA_CSV = 'metadata_with_transcript.csv'  # already includes correct paths
MODEL_OUTPUT = 'model/emotion_model.h5'
LABEL_ENCODER_OUTPUT = 'utils/label_encoder.pkl'

# --- Load dataset ---
df = pd.read_csv(DATA_CSV)

# --- Extract audio features (MFCC) ---
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

audio_features = []
for path in df['path']:
    audio_features.append(extract_audio_features(path))  # ✅ fixed here
audio_features = np.array(audio_features)

# --- Encode emotion labels ---
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['emotion'])
y = to_categorical(labels)

# --- Save label encoder ---
os.makedirs('utils', exist_ok=True)
with open(LABEL_ENCODER_OUTPUT, 'wb') as f:
    pickle.dump(label_encoder, f)

# --- Train/test split ---
X = audio_features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Build the model ---
model = Sequential()
model.add(Dense(256, input_shape=(X.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Train ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# --- Save model ---
os.makedirs('model', exist_ok=True)
model.save(MODEL_OUTPUT)
print("✅ Model trained and saved at:", MODEL_OUTPUT)
