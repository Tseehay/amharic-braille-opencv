"""
Braille Preprocessing Module
Handles image preprocessing for Amharic Braille detection including:
- Noise reduction
- Contrast enhancement
- Paper region detection
- Perspective correction
- Binarization
"""

import cv2
import numpy as np


class BraillePreprocessor:
    """Preprocesses images for Braille character detection."""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    
    def load_image(self, image_path):
        """Load an image from file path."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        return img
    
    def reduce_noise(self, image):
        """Apply noise reduction using Gaussian and bilateral filters."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply bilateral filter for edge-preserving smoothing
        filtered = cv2.bilateralFilter(smoothed, 20, 30, 30)
        
        return filtered
    
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE."""
        enhanced = self.clahe.apply(image)
        return enhanced
    
    def detect_paper_region(self, image):
        """
        Detect the paper region using edge detection and contour analysis.
        Returns the mask and largest contour of the paper.
        """
        # Edge detection
        edges = cv2.Canny(image, 100, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Get largest contour (paper)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        return mask, largest_contour
    
    def isolate_paper(self, image, mask):
        """
        Isolate paper region by masking out non-paper areas.
        Sets non-paper regions to white.
        """
        if len(image.shape) == 3:
            paper_only = cv2.bitwise_and(image, image, mask=mask)
            paper_only[np.where(mask == 0)] = [255, 255, 255]
        else:
            paper_only = cv2.bitwise_and(image, image, mask=mask)
            paper_only[np.where(mask == 0)] = 255
        
        return paper_only
    
    @staticmethod
    def order_points(pts):
        """
        Order points in a consistent manner:
        [top-left, top-right, bottom-right, bottom-left]
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum: top-left has smallest sum, bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Diff: top-right has smallest diff, bottom-left has largest diff
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def deskew_image(self, image, contour):
        """
        Apply perspective transformation to correct image skew.
        """
        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we don't have 4 points, return original image
        if len(approx) != 4:
            return image
        
        # Get source points
        src_pts = np.float32([approx[i][0] for i in range(4)])
        src_pts = self.order_points(src_pts)
        
        # Calculate destination dimensions
        width = max(
            np.linalg.norm(src_pts[0] - src_pts[1]),
            np.linalg.norm(src_pts[2] - src_pts[3])
        )
        height = max(
            np.linalg.norm(src_pts[0] - src_pts[3]),
            np.linalg.norm(src_pts[1] - src_pts[2])
        )
        
        # Define destination points
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Compute homography and apply transformation
        h_matrix, _ = cv2.findHomography(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, h_matrix, (int(width), int(height)))
        
        return warped
    
    def binarize(self, image):
        """
        Convert image to binary using adaptive thresholding.
        Inverted for braille dots (white on black).
        """
        binary = cv2.adaptiveThreshold(
            image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        return binary
    
    def preprocess(self, image_path):
        """
        Complete preprocessing pipeline.
        Returns preprocessed binary image ready for segmentation.
        """
        # Load image
        image = self.load_image(image_path)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Reduce noise
        denoised = self.reduce_noise(image)
        
        # Enhance contrast
        enhanced = self.enhance_contrast(denoised)
        
        # Detect paper region
        mask, contour = self.detect_paper_region(denoised)
        
        # Isolate paper
        if mask is not None:
            paper = self.isolate_paper(image, mask)
            if len(paper.shape) == 3:
                paper_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
            else:
                paper_gray = paper
        else:
            paper_gray = enhanced
            contour = None
        
        # Deskew if we have a valid contour
        if contour is not None:
            deskewed = self.deskew_image(paper_gray, contour)
        else:
            deskewed = paper_gray
        
        # Binarize
        binary = self.binarize(deskewed)
        
        return {
            'original': image,
            'gray': gray,
            'enhanced': enhanced,
            'deskewed': deskewed,
            'binary': binary
        }
