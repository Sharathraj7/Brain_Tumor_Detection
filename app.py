import os
import uuid
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
MODEL_PATH = 'trained_model/best.pt'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load your trained YOLO model
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Run YOLO inference
        results = model.predict(source=filepath, save=False)

        if not results or len(results[0].boxes) == 0:
            return jsonify({'message': 'No tumor detected'}), 200

        # Highest confidence detection
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().item()
        best_box = boxes[best_idx]

        cls_id = int(best_box.cls.item())
        conf = float(best_box.conf.item())
        label = model.names[cls_id]

        # Draw bounding box
        img = cv2.imread(filepath)
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save output image
        output_filename = "pred_" + unique_filename
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, img)

        # Return web-friendly path
        output_url = f"/{OUTPUT_FOLDER}/{output_filename}"

        return jsonify({
            'message': 'Prediction successful',
            'predicted_class': label,
            'confidence': round(conf, 2),
            'image_url': output_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
#ultralytics     8.3.202
#ultralytics-thop    2.0.17