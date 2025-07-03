from flask import Flask, request, jsonify, render_template
import whisper
import tempfile
from model_loader import predict_emotion
import os

app = Flask(__name__)
whisper_model = whisper.load_model("base")

EMOJI_MAP = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "surprised": "😲",
    "neutral": "😐",
    "fearful": "😨",
    "disgust": "🤢"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files["audio"]
    if not audio:
        return jsonify({"error": "No file uploaded"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.save(tmp.name)
        audio_path = tmp.name

    try:
        transcript = whisper_model.transcribe(audio_path)["text"]
        emotion = predict_emotion(audio_path, transcript)
        emoji = EMOJI_MAP.get(emotion.lower(), "😐")
        return jsonify({"transcript": transcript, "emotion": emotion, "emoji": emoji})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
