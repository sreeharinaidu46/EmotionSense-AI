import os
import csv
import whisper

AUDIO_DIR = "data/audio"  # Change if your audio is elsewhere
OUTPUT_CSV = "metadata_with_transcript.csv"

# Load whisper model
model = whisper.load_model("base")

# Collect all .wav audio files
audio_files = []
for root, _, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.lower().endswith(".wav"):
            full_path = os.path.join(root, file)
            audio_files.append(full_path)

print(f"Found {len(audio_files)} audio files.")

# Optional: derive emotion from filename or folder name
def extract_emotion_from_path(path):
    # Example: .../03-01-05-01-02-01-16.wav → map to label manually if needed
    # You can customize this logic based on folder/filename patterns
    parts = path.split("/")
    filename = parts[-1]
    emotion_code = filename.split("-")[2]
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }
    return emotion_map.get(emotion_code, "unknown")

# Write metadata
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["path", "emotion", "transcript"])

    for i, audio_path in enumerate(audio_files):
        try:
            result = model.transcribe(audio_path)
            text = result["text"]
            emotion = extract_emotion_from_path(audio_path)
            writer.writerow([audio_path, emotion, text])
            print(f"[{i+1}/{len(audio_files)}] ✅ {emotion} → {text}")
        except Exception as e:
            print(f"[{i+1}] ❌ Error processing {audio_path}: {e}")

print(f"\n✅ Transcriptions saved to: {OUTPUT_CSV}")
