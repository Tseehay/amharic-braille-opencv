"""
Utility functions for the Braille detection system.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def display_image(title, image, cmap='gray', figsize=(10, 8)):
    """
    Display an image using matplotlib.
    
    Args:
        title: Image title
        image: Image to display
        cmap: Colormap for grayscale images
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()


def save_image(image, path):
    """
    Save an image to disk.
    
    Args:
        image: Image to save
        path: Output path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def load_image(path):
    """
    Load an image from disk.
    
    Args:
        path: Image path
        
    Returns:
        Loaded image or None if not found
    """
    image = cv2.imread(path)
    if image is None:
        print(f"Warning: Could not load image from {path}")
    return image


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.avif')):
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory to search
        extensions: Tuple of valid image extensions
        
    Returns:
        List of image file paths
    """
    image_files = []
    
    if not os.path.exists(directory):
        return image_files
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(extensions):
            image_files.append(os.path.join(directory, filename))
    
    return sorted(image_files)


def visualize_pipeline_stages(stages_dict, save_path=None):
    """
    Visualize multiple processing stages in a grid.
    
    Args:
        stages_dict: Dictionary of {stage_name: image}
        save_path: Optional path to save the visualization
    """
    n_stages = len(stages_dict)
    n_cols = min(3, n_stages)
    n_rows = (n_stages + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_stages == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_stages > 1 else [axes]
    
    for idx, (stage_name, image) in enumerate(stages_dict.items()):
        ax = axes[idx]
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(stage_name)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_stages, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def create_text_from_predictions(predictions):
    """
    Create text string from prediction results.
    
    Args:
        predictions: List of (character, confidence) tuples
        
    Returns:
        Concatenated text string
    """
    text = ''.join([char for char, conf in predictions])
    return text


def calculate_average_confidence(predictions):
    """
    Calculate average confidence from predictions.
    
    Args:
        predictions: List of (character, confidence) tuples
        
    Returns:
        Average confidence value
    """
    if not predictions:
        return 0.0
    
    confidences = [conf for char, conf in predictions]
    return np.mean(confidences)


def format_results(predictions):
    """
    Format prediction results for display.
    
    Args:
        predictions: List of (character, confidence) tuples
        
    Returns:
        Formatted string
    """
    text = create_text_from_predictions(predictions)
    avg_confidence = calculate_average_confidence(predictions)
    
    result = f"Detected Text: {text}\n"
    result += f"Average Confidence: {avg_confidence:.2%}\n"
    result += f"Number of Characters: {len(predictions)}\n\n"
    result += "Character Details:\n"
    
    for idx, (char, conf) in enumerate(predictions, 1):
        result += f"  {idx}. '{char}' - {conf:.2%}\n"
    
    return result
