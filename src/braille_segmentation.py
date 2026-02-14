"""
Braille Segmentation Module
Handles segmentation of individual Braille characters from preprocessed images.
"""

import cv2
import numpy as np


class BrailleSegmenter:
    """Segments Braille characters from binary images."""
    
    def __init__(self, min_dot_size=5, dot_spacing_threshold=20, letter_spacing_threshold=20):
        """
        Initialize segmenter with configurable parameters.
        
        Args:
            min_dot_size: Minimum width/height for a valid Braille dot
            dot_spacing_threshold: Distance threshold to group dots into a row
            letter_spacing_threshold: Distance threshold to group rows into letters
        """
        self.min_dot_size = min_dot_size
        self.dot_spacing_threshold = dot_spacing_threshold
        self.letter_spacing_threshold = letter_spacing_threshold
    
    def detect_dots(self, binary_image):
        """
        Detect individual Braille dots using contour detection.
        
        Args:
            binary_image: Binary image with Braille dots (white dots on black)
            
        Returns:
            List of contours representing Braille dots
        """
        contours, _ = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > self.min_dot_size and h > self.min_dot_size:
                valid_contours.append(contour)
        
        return valid_contours
    
    def group_dots_into_rows(self, dot_contours):
        """
        Group Braille dots into rows based on y-coordinate proximity.
        
        Args:
            dot_contours: List of dot contours
            
        Returns:
            List of rows, where each row is a list of bounding boxes (x, y, w, h)
        """
        # Sort by y-coordinate, then x-coordinate
        sorted_contours = sorted(
            dot_contours, 
            key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0])
        )
        
        rows = []
        current_row = []
        previous_y = None
        
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h // 2
            
            if previous_y is None or abs(center_y - previous_y) < self.dot_spacing_threshold:
                current_row.append((x, y, w, h))
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [(x, y, w, h)]
            
            previous_y = center_y
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def group_rows_into_letters(self, rows):
        """
        Group rows of dots into individual letters based on x-coordinate proximity.
        
        Args:
            rows: List of rows from group_dots_into_rows
            
        Returns:
            List of letters, where each letter is a list of bounding boxes
        """
        letters = []
        
        for row in rows:
            # Sort by x-coordinate
            row = sorted(row, key=lambda b: b[0])
            current_letter = []
            previous_x = None
            
            for bbox in row:
                x, y, w, h = bbox
                center_x = x + w // 2
                
                if previous_x is None or abs(center_x - previous_x) < self.letter_spacing_threshold:
                    current_letter.append(bbox)
                else:
                    if current_letter:
                        letters.append(current_letter)
                    current_letter = [bbox]
                
                previous_x = center_x
            
            if current_letter:
                letters.append(current_letter)
        
        return letters
    
    def extract_letter_images(self, image, letters):
        """
        Extract individual letter images from the original image.
        
        Args:
            image: Original grayscale or binary image
            letters: List of letters from group_rows_into_letters
            
        Returns:
            List of cropped letter images with their bounding boxes
        """
        letter_images = []
        
        for letter in letters:
            # Find bounding box for the entire letter
            min_x = min([x for x, y, w, h in letter])
            min_y = min([y for x, y, w, h in letter])
            max_x = max([x + w for x, y, w, h in letter])
            max_y = max([y + h for x, y, w, h in letter])
            
            # Add padding
            padding = 5
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(image.shape[1], max_x + padding)
            max_y = min(image.shape[0], max_y + padding)
            
            # Crop letter region
            letter_image = image[min_y:max_y, min_x:max_x]
            
            letter_images.append({
                'image': letter_image,
                'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                'dots': letter
            })
        
        return letter_images
    
    def segment(self, binary_image, original_image=None):
        """
        Complete segmentation pipeline.
        
        Args:
            binary_image: Binary image with Braille dots
            original_image: Optional original grayscale image for extraction
            
        Returns:
            Dictionary containing segmentation results
        """
        if original_image is None:
            original_image = binary_image
        
        # Detect dots
        dot_contours = self.detect_dots(binary_image)
        
        # Group into rows
        rows = self.group_dots_into_rows(dot_contours)
        
        # Group into letters
        letters = self.group_rows_into_letters(rows)
        
        # Extract letter images
        letter_images = self.extract_letter_images(original_image, letters)
        
        return {
            'dot_contours': dot_contours,
            'rows': rows,
            'letters': letters,
            'letter_images': letter_images
        }
    
    def visualize_segmentation(self, image, segmentation_result):
        """
        Visualize the segmentation results on the image.
        
        Args:
            image: Image to draw on
            segmentation_result: Result from segment() method
            
        Returns:
            Image with visualization overlay
        """
        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes for each letter
        for letter_info in segmentation_result['letter_images']:
            x, y, w, h = letter_info['bbox']
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return output
