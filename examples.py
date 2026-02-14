#!/usr/bin/env python3
"""
Example script demonstrating how to use the Amharic Braille Detection System.
"""

import sys
sys.path.insert(0, 'src')

from braille_preprocessing import BraillePreprocessor
from braille_segmentation import BrailleSegmenter
from braille_classifier import BrailleClassifier
from main import BrailleDetector
import cv2


def example_1_basic_usage():
    """Example 1: Basic usage with BrailleDetector"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize detector
    detector = BrailleDetector()
    
    # Process a single image
    result = detector.process_image('raw_dataset/img5.jpg')
    
    # Display results
    print(f"Detected {result['num_characters']} characters")
    if result['predictions']:
        text = ''.join([char for char, conf in result['predictions']])
        print(f"Text: {text}")


def example_2_individual_components():
    """Example 2: Using individual components"""
    print("\n" + "=" * 60)
    print("Example 2: Using Individual Components")
    print("=" * 60)
    
    # Initialize components
    preprocessor = BraillePreprocessor()
    segmenter = BrailleSegmenter(
        min_dot_size=5,
        dot_spacing_threshold=20,
        letter_spacing_threshold=20
    )
    classifier = BrailleClassifier()
    
    # Process step by step
    print("\n1. Preprocessing...")
    preprocessed = preprocessor.preprocess('raw_dataset/img5.jpg')
    print(f"   Binary image shape: {preprocessed['binary'].shape}")
    
    print("\n2. Segmentation...")
    segmentation = segmenter.segment(
        preprocessed['binary'],
        preprocessed['deskewed']
    )
    print(f"   Detected {len(segmentation['dot_contours'])} dots")
    print(f"   Extracted {len(segmentation['letter_images'])} characters")
    
    print("\n3. Classification...")
    predictions = []
    for idx, letter_info in enumerate(segmentation['letter_images']):
        char, confidence = classifier.predict(letter_info['image'])
        predictions.append((char, confidence))
        print(f"   Character {idx+1}: '{char}' (confidence: {confidence:.2%})")


def example_3_batch_processing():
    """Example 3: Batch processing"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    # Initialize detector
    detector = BrailleDetector()
    
    # Process all images in a directory
    results = detector.process_batch(
        input_dir='raw_dataset',
        output_dir='output/examples',
        save_intermediates=False
    )
    
    print(f"\nProcessed {len(results)} images")
    for filename, result in results.items():
        if 'error' not in result:
            num_chars = result.get('num_characters', 0)
            print(f"  {filename}: {num_chars} characters")


def example_4_custom_parameters():
    """Example 4: Custom preprocessing and segmentation parameters"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Parameters")
    print("=" * 60)
    
    # Custom preprocessor
    preprocessor = BraillePreprocessor()
    
    # Custom segmenter with different thresholds
    segmenter = BrailleSegmenter(
        min_dot_size=3,  # Detect smaller dots
        dot_spacing_threshold=25,  # Wider spacing for rows
        letter_spacing_threshold=30  # Wider spacing for letters
    )
    
    preprocessed = preprocessor.preprocess('raw_dataset/img9.jpg')
    segmentation = segmenter.segment(
        preprocessed['binary'],
        preprocessed['deskewed']
    )
    
    print(f"Custom parameters detected {len(segmentation['letter_images'])} characters")


def example_5_save_intermediates():
    """Example 5: Saving intermediate processing stages"""
    print("\n" + "=" * 60)
    print("Example 5: Saving Intermediate Results")
    print("=" * 60)
    
    detector = BrailleDetector()
    
    result = detector.process_image(
        'raw_dataset/img5.jpg',
        save_intermediates=True,
        output_dir='output/example_5'
    )
    
    print("Saved files:")
    print("  - Gray image")
    print("  - Enhanced image")
    print("  - Deskewed image")
    print("  - Binary image")
    print("  - Segmentation visualization")
    print("  - Individual letter images")
    print("\nAll files saved to output/example_5/")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Amharic Braille Detection - Examples")
    print("=" * 60)
    
    # Run all examples
    example_1_basic_usage()
    example_2_individual_components()
    example_3_batch_processing()
    example_4_custom_parameters()
    example_5_save_intermediates()
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
