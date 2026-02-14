"""
Braille Classifier Module
Handles classification of segmented Braille characters using a CNN model.
"""

import cv2
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers


class BrailleClassifier:
    """
    CNN-based classifier for Amharic Braille characters.
    """
    
    # Amharic Braille character mapping (example set - can be extended)
    # This is a simplified mapping for demonstration
    BRAILLE_CHARACTERS = [
        'ሀ', 'ለ', 'ሐ', 'መ', 'ሠ', 'ረ', 'ሰ', 'ሸ', 'ቀ', 'በ',
        'ተ', 'ቸ', 'ኀ', 'ነ', 'ኘ', 'አ', 'ከ', 'ኸ', 'ወ', 'ዐ',
        'ዘ', 'ዠ', 'የ', 'ደ', 'ጀ', 'ገ', 'ጠ', 'ጨ', 'ጰ', 'ጸ',
        'ፀ', 'ፈ', 'ፐ', ' ', '.', ',', '?', '!'
    ]
    
    def __init__(self, model_path=None, input_shape=(64, 64, 1)):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to pre-trained model file
            input_shape: Input shape for the model
        """
        self.input_shape = input_shape
        self.num_classes = len(self.BRAILLE_CHARACTERS)
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """
        Build a CNN model for Braille character classification.
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image):
        """
        Preprocess a single letter image for classification.
        
        Args:
            image: Input image (grayscale or binary)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Add channel dimension
        if len(normalized.shape) == 2:
            normalized = np.expand_dims(normalized, axis=-1)
        
        return normalized
    
    def predict(self, image):
        """
        Predict the character for a single image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (predicted_character, confidence)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Build or load a model first.")
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Add batch dimension
        batch = np.expand_dims(processed, axis=0)
        
        # Predict
        predictions = self.model.predict(batch, verbose=0)
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_char = self.BRAILLE_CHARACTERS[predicted_idx]
        
        return predicted_char, confidence
    
    def predict_batch(self, images):
        """
        Predict characters for a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of (predicted_character, confidence) tuples
        """
        if self.model is None:
            raise ValueError("Model not initialized. Build or load a model first.")
        
        # Preprocess all images
        processed_images = []
        for img in images:
            processed = self.preprocess_image(img)
            processed_images.append(processed)
        
        # Stack into batch
        batch = np.stack(processed_images, axis=0)
        
        # Predict
        predictions = self.model.predict(batch, verbose=0)
        
        # Extract results
        results = []
        for pred in predictions:
            predicted_idx = np.argmax(pred)
            confidence = pred[predicted_idx]
            predicted_char = self.BRAILLE_CHARACTERS[predicted_idx]
            results.append((predicted_char, confidence))
        
        return results
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, callbacks=None):
        """
        Train the model on labeled data.
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, path):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a model from disk."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': results[0],
            'accuracy': results[1]
        }
