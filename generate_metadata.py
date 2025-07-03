import os
import csv

# Path to your audio data
AUDIO_DIR = "data/audio"

# RAVDESS emotion label mapping (3rd number in filename)
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_emotion_from_filename(filename):
    """
    Extract emotion code from RAVDESS filename (format: xx-xx-EMOTION-xx-xx-xx-xx.wav)
    """
    try:
        emotion_code = filename.split("-")[2]
        return EMOTION_MAP.get(emotion_code, "unknown")
    except IndexError:
        return "unknown"

def generate_metadata():
    metadata = []
    for root, dirs, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=".")
                emotion = extract_emotion_from_filename(file)
                metadata.append([rel_path, emotion])

    # Save to CSV
    with open("metadata.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "emotion"])
        writer.writerows(metadata)

    print("✅ Metadata generated and saved to metadata.csv")

if __name__ == "__main__":
    generate_metadata()
