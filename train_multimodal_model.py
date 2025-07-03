import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt

CSV_PATH = "metadata_with_transcript.csv"
MODEL_SAVE_PATH = "model/multimodal_model"
BEST_MODEL_PATH = "model/best_model.h5"

# Load data
df = pd.read_csv(CSV_PATH)

# Feature extraction
X_audio = []
X_text = []
y = []

for index, row in df.iterrows():
    y_audio, sr = librosa.load(row["path"], sr=None)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    X_audio.append(mfcc_mean)
    X_text.append(row["transcript"])
    y.append(row["emotion"])

X_audio = np.array(X_audio)

# Preprocessing
scaler = StandardScaler()
X_audio_scaled = scaler.fit_transform(X_audio)

vectorizer = TfidfVectorizer(max_features=300)
X_text_vectorized = vectorizer.fit_transform(X_text).toarray()

text_input_dim = X_text_vectorized.shape[1]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save encoders and scalers
os.makedirs("data", exist_ok=True)
joblib.dump(scaler, "data/audio_scaler.pkl")
joblib.dump(vectorizer, "data/text_vectorizer.pkl")
joblib.dump(label_encoder, "data/label_encoder.pkl")

# Train-test split
X_audio_train, X_audio_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
    X_audio_scaled, X_text_vectorized, y_encoded, test_size=0.2, random_state=42
)

# Define model
input_audio = Input(shape=(40,), name="audio_input")
a1 = Dense(128, activation="relu")(input_audio)
a1 = BatchNormalization()(a1)
a1 = Dropout(0.4)(a1)

a2 = Dense(64, activation="relu")(a1)
a2 = Dropout(0.3)(a2)

input_text = Input(shape=(text_input_dim,), name="text_input")
t1 = Dense(128, activation="relu")(input_text)
t1 = BatchNormalization()(t1)
t1 = Dropout(0.4)(t1)

t2 = Dense(64, activation="relu")(t1)
t2 = Dropout(0.3)(t2)

concat = Concatenate()([a2, t2])
x = Dense(64, activation="relu")(concat)
x = Dropout(0.3)(x)

output = Dense(len(label_encoder.classes_), activation="softmax")(x)

model = Model(inputs=[input_audio, input_text], outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Callbacks
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]

# Train
history = model.fit(
    [X_audio_train, X_text_train], y_train,
    validation_data=([X_audio_val, X_text_val], y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# Save final model
model.save(MODEL_SAVE_PATH)

# Plot accuracy
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/training_accuracy.png")
