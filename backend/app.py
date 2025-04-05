from flask import Flask, request, jsonify
import joblib
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import os
import numpy as np
import librosa
import joblib
import soundfile as sf
import noisereduce as nr
import python_speech_features as psf
import scipy.signal

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")

# Function to compute RPDE (Recurrence Period Density Entropy)
def compute_rpde(y):
    try:
        _, _, Sxx = scipy.signal.spectrogram(y)
        entropy = -np.sum(Sxx * np.log2(Sxx + 1e-10)) / Sxx.shape[1]
        return entropy
    except:
        return 0

# Function to compute PPE (Pitch Period Entropy)
def compute_ppe(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    if len(pitches) > 0:
        return np.std(pitches) / np.mean(pitches)
    return 0

# Function to compute GTCC (Gammatone Cepstral Coefficients)
def compute_gtcc(y, sr):
    gtcc = psf.mfcc(y, samplerate=sr, numcep=13)
    return np.mean(gtcc, axis=0)

# Function to compute DFA (Detrended Fluctuation Analysis)
def compute_dfa(y):
    return np.std(np.diff(y))

# Function to preprocess and extract features
def extract_features(file_path):
    try:
        # Load and preprocess audio
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        y_denoised = np.nan_to_num(y_denoised)  # Handle NaN/Inf values

        # Extract features
        mfcc = np.mean(librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=13), axis=1)
        rpde = compute_rpde(y_denoised)
        ppe = compute_ppe(y_denoised, sr)
        gtcc = compute_gtcc(y_denoised, sr)
        dfa = compute_dfa(y_denoised)

        # Flatten GTCC features
        gtcc_features = gtcc.tolist() if isinstance(gtcc, np.ndarray) else [0] * 13
        
        # Create feature vector
        features = np.array(mfcc.tolist() + [rpde, ppe] + gtcc_features + [dfa])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

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