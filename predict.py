import numpy as np
import cv2
import tensorflow as tf
import argparse

class PneumoniaPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        
    def preprocess_image(self, image_path, img_size=(224, 224)):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, image_path):
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img)[0][0]
        
        class_names = ['Normal', 'Pneumonia']
        predicted_class = class_names[1 if prediction > 0.5 else 0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'Normal': 1 - prediction,
                'Pneumonia': prediction
            }
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='Path to chest X-ray image')
    parser.add_argument('--model_path', default='models/saved_models/pneumonia_detection_model.h5')
    args = parser.parse_args()
    
    predictor = PneumoniaPredictor(args.model_path)
    result = predictor.predict(args.image_path)
    
    print(f"Prediction: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nProbabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.2%}")

if __name__ == "__main__":
    main()
