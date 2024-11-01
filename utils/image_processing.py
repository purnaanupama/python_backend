import cv2
import numpy as np
import pytesseract
from sklearn.cluster import DBSCAN

# Set the Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust if needed

def process_image(image):
    output_image = image.copy()  # Copy for drawing detected marks

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to highlight edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=5, minLineLength=10, maxLineGap=5)

    # Variables to store potential X marks
    x_lines = []
    detected_centers = []

    # Adjustable box size for contour scanning
    box_width = 80
    box_height = 260
    box_offset_y = 32

    # Identify lines that might form X shapes based on angle
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle)

            if 20 <= angle <= 70 or 110 <= angle <= 160:
                x_lines.append((x1, y1, x2, y2))

    # Detect intersections to identify potential X marks
    for i in range(len(x_lines)):
        for j in range(i + 1, len(x_lines)):
            line1 = x_lines[i]
            line2 = x_lines[j]

            if (line1[0] < line2[2] and line1[2] > line2[0]) and (line1[1] < line2[3] and line1[3] > line2[1]):
                intersection_x = int((line1[0] + line1[2] + line2[0] + line2[2]) / 4)
                intersection_y = int((line1[1] + line1[3] + line2[1] + line2[3]) / 4)
                detected_centers.append((intersection_x, intersection_y))

    detected_centers = np.array(detected_centers)

    if len(detected_centers) > 0:
        dbscan = DBSCAN(eps=30, min_samples=1).fit(detected_centers)
        unique_clusters = np.unique(dbscan.labels_)
        x_marks_count = len(unique_clusters)
        candidate_name = ""
        additional_x_found = False
        detected_texts_in_area = []

        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        for cluster_id in unique_clusters:
            cluster_points = detected_centers[dbscan.labels_ == cluster_id]
            center_x, center_y = np.mean(cluster_points, axis=0).astype(int)

            box_top_left = (center_x - box_width // 2, center_y + box_offset_y)
            box_bottom_right = (center_x + box_width // 2, center_y + box_offset_y + box_height)
            roi = gray[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]

            roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            roi_thresh = cv2.adaptiveThreshold(roi_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 0.5 < aspect_ratio < 2.5:
                    additional_x_found = True
                    break

            thickness = 30
            for i in range(len(data['text'])):
                text = data['text'][i]
                if text.strip():
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    left_area = (0, center_y - thickness, center_x, center_y + thickness)
                    right_area = (center_x, center_y - thickness, output_image.shape[1], center_y + thickness)

                    if ((x + w > left_area[0] and x < left_area[2]) and (y + h > left_area[1] and y < left_area[3])) or \
                       ((x + w > right_area[0] and x < right_area[2]) and (y + h > right_area[1] and y < right_area[3])):
                        detected_texts_in_area.append(text)
                        cv2.putText(output_image, text, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

            candidate_text = pytesseract.image_to_string(roi_thresh, config="--psm 6").strip()
            if candidate_text:
                candidate_name = candidate_text

            cv2.circle(output_image, (center_x, center_y), 5, (0, 0, 255), -1)

        if additional_x_found:
            result = {"status": "rejected", "candidate_name": candidate_name, "additional_x_found": True, "x_marks_count": x_marks_count, "detected_texts": detected_texts_in_area}
        elif x_marks_count == 1:
            result = {"status": "accepted", "candidate_name": candidate_name, "additional_x_found": False, "x_marks_count": x_marks_count, "detected_texts": detected_texts_in_area}
        else:
            result = {"status": "no_valid_x", "detected_texts": detected_texts_in_area}
    else:
        result = {"status": "no_x_detected"}

    return result, output_image
