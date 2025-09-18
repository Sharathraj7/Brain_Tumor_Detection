import os
import uuid
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

# Create Flask app
app = Flask(__name__)

# Deployment-friendly paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model', 'best.pt')

# Create directories if they don’t exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model once (important for deployment)
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Save file with unique name
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Run YOLO prediction (save output image)
        results = model.predict(source=filepath, save=True, project=OUTPUT_FOLDER, name=unique_filename)

        # Get output directory (YOLO saves under OUTPUT_FOLDER/name)
        output_dir = os.path.join(OUTPUT_FOLDER, unique_filename)
        output_files = list(results[0].save_dir.glob("*.jpg"))

        if output_files:
            output_url = os.path.join('static', 'outputs', unique_filename, output_files[0].name)
            return jsonify({'message': 'Prediction successful', 'image_url': output_url})
        else:
            return jsonify({'error': 'No output image generated'}), 500

    return jsonify({'error': 'Unknown error occurred'}), 500


if __name__ == '__main__':
    # For deployment: bind to PORT env variable if available
    port = int(os.environ.get("PORT", 7860))  # 7860 works on Hugging Face
    app.run(host="0.0.0.0", port=port)
