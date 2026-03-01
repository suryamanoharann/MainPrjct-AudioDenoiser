import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torchaudio.models import HDemucs

# Configure Flask to serve your 'frontend' folder automatically
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# --- Configuration & Folders ---
UPLOAD_FOLDER = 'uploads'
STEMS_FOLDER = 'stems' # We'll keep generated audio outside the frontend folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STEMS_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEMS = ['Speech', 'Music', 'Impacts', 'Alerts', 'Environmental', 'Mechanical']
MODEL_PATH = "MAINPRJCT_21epoch.pth"

# --- Initialize 6-Stem HDemucs Model ---
print(f"Loading HDemucs Model on {DEVICE}...")
model = HDemucs(sources=STEMS, audio_channels=2)

if os.path.exists(MODEL_PATH):
    # Matches training loop: model.module.state_dict() saved as 'model'
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    print("✅ Model loaded successfully.")
else:
    print(f"⚠️ WARNING: {MODEL_PATH} not found. Using untrained weights.")

model = model.to(DEVICE)
model.eval()

# --- Helper: RMS Energy ---
def calculate_rms(tensor_audio):
    """Calculates energy to tell the JS frontend which sliders to render."""
    audio_np = tensor_audio.detach().cpu().numpy()
    return float(np.sqrt(np.mean(audio_np**2)))

# --- Routing ---
@app.route('/')
def index():
    # Serves frontend/index.html by default
    return app.send_static_file('index.html')

@app.route('/stems/<path:filename>')
def serve_stem(filename):
    # Safely serves the generated .wav files from the root stems folder
    return send_from_directory(STEMS_FOLDER, filename)

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['audio_file']
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    try:
        # 1. Load and prep audio
        wav, sr = torchaudio.load(input_path)
        
        if sr != 16000:
            wav = T.Resample(sr, 16000)(wav)
            
        if wav.shape == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape > 2:
            wav = wav[:2, :]
            
        wav = wav.unsqueeze(0).to(DEVICE) # Shape: [1, 2, length]

        # 2. Run PyTorch Inference
        print("Running PyTorch Inference...")
        with torch.no_grad():
            estimates = model(wav) # Shape: [1, 6, 2, length]

        # 3. Process and Save Stems
        response_data = []
        
        for i, stem_name in enumerate(STEMS):
            stem_id = stem_name.lower()
            stem_tensor = estimates[0, i, :, :] 
            
            rms_val = calculate_rms(stem_tensor)
            
            # Save file
            output_filename = f"{stem_id}.wav"
            output_path = os.path.join(STEMS_FOLDER, output_filename)
            torchaudio.save(output_path, stem_tensor.cpu(), 16000)
            
            response_data.append({
                'id': stem_id,
                'name': stem_name,
                'rmsEnergy': rms_val,
                'url': f'/stems/{output_filename}'
            })

        os.remove(input_path)
        return jsonify({'stems': response_data}), 200

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)