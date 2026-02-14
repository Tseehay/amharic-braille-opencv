"""
Main Braille Detection Pipeline
Integrates preprocessing, segmentation, and classification for end-to-end Braille detection.
"""

import cv2
import os
import argparse
from .braille_preprocessing import BraillePreprocessor
from .braille_segmentation import BrailleSegmenter
from .braille_classifier import BrailleClassifier
from .utils import save_image, load_image, get_image_files, format_results, ensure_dir


class BrailleDetector:
    """
    Complete Braille detection pipeline.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the Braille detector with all components.
        
        Args:
            model_path: Path to pre-trained classification model
        """
        self.preprocessor = BraillePreprocessor()
        self.segmenter = BrailleSegmenter()
        self.classifier = BrailleClassifier(model_path=model_path)
    
    def process_image(self, image_path, save_intermediates=False, output_dir=None):
        """
        Process a single Braille image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            save_intermediates: Whether to save intermediate processing stages
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing detection results
        """
        # Preprocess
        preprocessed = self.preprocessor.preprocess(image_path)
        
        # Segment
        segmentation = self.segmenter.segment(
            preprocessed['binary'], 
            preprocessed['deskewed']
        )
        
        # Classify each segmented character
        predictions = []
        if segmentation['letter_images']:
            for letter_info in segmentation['letter_images']:
                letter_image = letter_info['image']
                char, confidence = self.classifier.predict(letter_image)
                predictions.append((char, confidence))
        
        # Create visualization
        visualization = self.segmenter.visualize_segmentation(
            preprocessed['deskewed'],
            segmentation
        )
        
        # Save intermediate results if requested
        if save_intermediates and output_dir:
            ensure_dir(output_dir)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            save_image(preprocessed['gray'], 
                      os.path.join(output_dir, f"{base_name}_1_gray.jpg"))
            save_image(preprocessed['enhanced'], 
                      os.path.join(output_dir, f"{base_name}_2_enhanced.jpg"))
            save_image(preprocessed['deskewed'], 
                      os.path.join(output_dir, f"{base_name}_3_deskewed.jpg"))
            save_image(preprocessed['binary'], 
                      os.path.join(output_dir, f"{base_name}_4_binary.jpg"))
            save_image(visualization, 
                      os.path.join(output_dir, f"{base_name}_5_segmentation.jpg"))
            
            # Save individual letters
            letters_dir = os.path.join(output_dir, f"{base_name}_letters")
            ensure_dir(letters_dir)
            for idx, letter_info in enumerate(segmentation['letter_images']):
                save_image(letter_info['image'], 
                          os.path.join(letters_dir, f"letter_{idx+1}.jpg"))
        
        return {
            'predictions': predictions,
            'num_characters': len(predictions),
            'preprocessed': preprocessed,
            'segmentation': segmentation,
            'visualization': visualization
        }
    
    def process_batch(self, input_dir, output_dir, save_intermediates=False):
        """
        Process multiple images from a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            save_intermediates: Whether to save intermediate stages
            
        Returns:
            Dictionary mapping image names to results
        """
        ensure_dir(output_dir)
        
        image_files = get_image_files(input_dir)
        results = {}
        
        print(f"Processing {len(image_files)} images...")
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"\nProcessing {idx}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                result = self.process_image(
                    image_path,
                    save_intermediates=save_intermediates,
                    output_dir=output_dir
                )
                
                results[os.path.basename(image_path)] = result
                
                # Print predictions
                if result['predictions']:
                    text = ''.join([char for char, conf in result['predictions']])
                    print(f"  Detected: {text}")
                else:
                    print("  No characters detected")
                    
            except Exception as e:
                print(f"  Error processing {image_path}: {str(e)}")
                results[os.path.basename(image_path)] = {'error': str(e)}
        
        return results


def main():
    """Command-line interface for the Braille detector."""
    parser = argparse.ArgumentParser(
        description='Amharic Braille Detection System'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image to process'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple images in batch mode'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='raw_dataset',
        help='Input directory for batch processing (default: raw_dataset)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to pre-trained model file'
    )
    
    parser.add_argument(
        '--save-intermediates',
        action='store_true',
        help='Save intermediate processing stages'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    print("Initializing Braille Detector...")
    detector = BrailleDetector(model_path=args.model)
    
    if args.batch:
        # Batch processing
        print(f"\nBatch processing images from {args.input_dir}")
        results = detector.process_batch(
            args.input_dir,
            args.output_dir,
            save_intermediates=args.save_intermediates
        )
        
        print(f"\n{'='*60}")
        print("Batch Processing Complete!")
        print(f"Processed {len(results)} images")
        print(f"Results saved to {args.output_dir}")
        
    elif args.image:
        # Single image processing
        print(f"\nProcessing single image: {args.image}")
        result = detector.process_image(
            args.image,
            save_intermediates=args.save_intermediates,
            output_dir=args.output_dir
        )
        
        print(f"\n{'='*60}")
        print("Results:")
        print(format_results(result['predictions']))
        
        if args.save_intermediates:
            print(f"Intermediate results saved to {args.output_dir}")
    
    else:
        parser.print_help()
        print("\nPlease specify either --image or --batch mode.")


if __name__ == '__main__':
    main()
