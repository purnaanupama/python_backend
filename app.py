from flask import Flask, request, jsonify
import cv2
import numpy as np
from utils.image_processing import process_image
from utils.image_processing2 import NumberVoteDetector

app = Flask(__name__)
detector = NumberVoteDetector()

#vote for X detection
@app.route('/process-vote-presidential', methods=['POST'])
def process_vote1():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    result, _ = process_image(image)
    return jsonify(result)

#route for 123 vote
@app.route('/process-ballot', methods=['POST'])
def process_ballot_route():
    if 'file' not in request.files:
        return jsonify({'status': 'Error', 'message': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'Error', 'message': 'No selected file'}), 400

    image_data = file.read()  # Read the image file from the request
    results = detector.process_ballot(image_data) 
    return jsonify(results)  # Return the response as JSON

if __name__ == '__main__':
    app.run(debug=True)
