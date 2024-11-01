import cv2
import numpy as np
import pytesseract
import logging
from typing import List, Dict

class NumberVoteDetector:
    def __init__(self, candidate_section_ratio: float = 0.69, vote_section_start_ratio: float = 0.25):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.custom_config_digits = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        self.custom_config_text = r'--oem 3 --psm 6'
        self.candidate_section_ratio = candidate_section_ratio
        self.vote_section_start_ratio = vote_section_start_ratio

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return cleaned

    def is_valid_ballot(self, image: np.ndarray) -> bool:
        self.logger.info("Validating ballot paper structure")
        return True  # Assume it's valid for this example

    def detect_votes(self, processed_image: np.ndarray) -> List[Dict[str, int]]:
        self.logger.info("Detecting votes in the ballot paper")
        
        height, width = processed_image.shape
        candidate_section_end = int(width * self.candidate_section_ratio)
        vote_section_start = int(width * self.vote_section_start_ratio)

        votes = []
        row_height = height // 6  # Assuming 6 rows of candidates

        for i in range(6):  # Loop through each candidate row
            y_start = i * row_height
            y_end = (i + 1) * row_height

            candidate_roi = processed_image[y_start:y_end, :candidate_section_end]
            vote_roi = processed_image[y_start:y_end, vote_section_start:]

            candidate_name = pytesseract.image_to_string(candidate_roi, config=self.custom_config_text).strip()
            vote_number_text = pytesseract.image_to_string(vote_roi, config=self.custom_config_digits).strip()

            if candidate_name and vote_number_text.isdigit():
                votes.append({
                    'candidate': candidate_name,
                    'vote_number': int(vote_number_text)
                })

        self.logger.info(f"Detected votes: {votes}")
        return votes

    def process_ballot(self, image_data: bytes) -> Dict:
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.logger.error("Failed to decode image from request data")
            return {'status': 'Error', 'message': 'Invalid image data'}

        processed = self.preprocess_image(image)
        
        if not self.is_valid_ballot(processed):
            return {'status': 'Rejected', 'message': 'Invalid ballot paper'}

        votes = self.detect_votes(processed)
        return {
            'status': 'Accepted' if votes else 'Rejected',
            'votes': votes
        }
