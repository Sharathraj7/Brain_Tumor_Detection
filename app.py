import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

# ------------------- Flask App ------------------- #
app = Flask(__name__)

# ------------------- Paths ------------------- #
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'static' / 'outputs'
MODEL_PATH = BASE_DIR / 'trained_model' / 'best.pt'

# Create folders if not exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ------------------- Load YOLO Model ------------------- #
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ------------------- Routes ------------------- #
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save uploaded file with unique name
        unique_filename = str(uuid.uuid4()) + Path(file.filename).suffix
        filepath = UPLOAD_FOLDER / unique_filename
        file.save(filepath)

        # Run YOLO prediction
        results = model.predict(source=str(filepath), save=True, project=str(OUTPUT_FOLDER), name=unique_filename)

        # Get output image path
        output_dir = OUTPUT_FOLDER / unique_filename
        output_files = list(output_dir.glob("*.jpg"))

        if output_files:
            output_url = f"/static/outputs/{unique_filename}/{output_files[0].name}"
            return jsonify({'message': 'Prediction successful', 'image_url': output_url})
        else:
            return jsonify({'error': 'No output image generated'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ------------------- Run App ------------------- #
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)
