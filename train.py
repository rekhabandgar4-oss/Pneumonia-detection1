import os
import yaml
import tensorflow as tf
from src.data.data_loader import DataLoader
from src.data.preprocessing import Preprocessor
from src.models.cnn_model import CNNModel

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    data_loader = DataLoader(config)
    preprocessor = Preprocessor(config)
    
    # Load and preprocess data
    print("Loading data...")
    images, labels = data_loader.load_data('data/raw/train')
    
    print("Preprocessing data...")
    images = preprocessor.normalize_images(images)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.split_data(images, labels)
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = data_loader.create_dataset(X_train, y_train, augment=True)
    val_dataset = data_loader.create_dataset(X_val, y_val, augment=False)
    
    # Build model
    print("Building model...")
    model_builder = CNNModel(config)
    model = model_builder.build_custom_cnn()
    model = model_builder.compile_model(model)
    
    # Print model summary
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['training']['epochs'],
        verbose=1
    )
    
    # Save model
    os.makedirs('models/saved_models', exist_ok=True)
    model.save('models/saved_models/pneumonia_detection_model.h5')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
