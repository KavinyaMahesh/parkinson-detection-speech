from flask import Flask, request, jsonify
import joblib
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    y = nr.reduce_noise(y=y, sr=sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    return np.hstack(mfcc).reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    file_path = "temp.wav"
    file.save(file_path)
    features = extract_features(file_path)
    features = scaler.transform(features)
    prediction = model.predict(features)[0]
    result = "Parkinson" if prediction == 1 else "Healthy"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)