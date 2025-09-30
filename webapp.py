import argparse
import os
import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import defaultdict
import time

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("last.pt")  # Ensure 'last.pt' is in the same directory or provide the correct path

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_video():
    if request.method == "POST":
        if 'file' in request.files:
            # Save uploaded video file
            uploaded_file = request.files['file']
            filename = secure_filename(uploaded_file.filename)
            upload_folder = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, filename)
            uploaded_file.save(filepath)

            # Open video file using OpenCV
            cap = cv2.VideoCapture(filepath)

            if not cap.isOpened():
                return render_template('index.html', message="Could not open video file.")

            # Set frame rate to 1 FPS
            frame_rate = 1
            prev_time = 0

            # Initialize class counts
            class_counts = defaultdict(int)
            sanitized = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Get current time
                current_time = time.time()

                # Process frame every 1 second (1 FPS)
                if current_time - prev_time >= 1.0 / frame_rate:
                    prev_time = current_time

                    # Perform inference
                    results = model.predict(source=frame, save=False)  # Predict on the frame

                    # Analyze results
                    for result in results:
                        names_dict = result.names
                        probs = result.probs.data.tolist()

                        # Filter predictions with probability greater than 0.85
                        high_confidence_classes = [names_dict[i] for i, prob in enumerate(probs) if prob > 0.85]

                        # Count high-confidence predictions
                        for class_name in high_confidence_classes:
                            class_counts[class_name] += 1

            cap.release()

            # Check if all class counts are more than twice
            sanitized = all(count > 2 for count in class_counts.values())

            # Generate appropriate message
            message = "Not Sanitized Properly" if sanitized else "Sanitized Properly"
            return render_template('index.html', message=message)

    return render_template('index.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv8 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
