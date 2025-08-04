import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data():
    """Load and preprocess the MNIST dataset from CSV"""
    print("Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv('train.csv/train.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Unique labels: {sorted(df['label'].unique())}")
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Reshape X to 28x28 images
    X = X.reshape(-1, 28, 28, 1)
    
    # Normalize pixel values to [0, 1]
    X = X.astype('float32') / 255.0
    
    # Convert labels to categorical
    y = keras.utils.to_categorical(y, 10)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val

def create_data_augmentation():
    """Create data augmentation pipeline"""
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    return data_augmentation

def create_enhanced_model():
    """Create an enhanced CNN model for MNIST classification"""
    
    # Data augmentation
    data_augmentation = create_data_augmentation()
    
    model = keras.Sequential([
        # Data augmentation
        data_augmentation,
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_mnist_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('enhanced_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_val, y_val):
    """Evaluate the model and create visualizations"""
    # Make predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Enhanced Model - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"\nFinal Validation Accuracy: {accuracy:.4f}")
    
    return accuracy

def create_model_for_streamlit(model, X_train):
    """Create a simplified model compatible with the current Streamlit app"""
    # Create a new model that takes flattened input (784 features)
    streamlit_model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Reshape((28, 28, 1)),
        # Copy the layers from the trained model (excluding data augmentation)
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Copy weights from the trained model (excluding data augmentation layer)
    trained_layers = model.layers[1:]  # Skip data augmentation layer
    streamlit_layers = streamlit_model.layers[1:]  # Skip reshape layer
    
    for trained_layer, streamlit_layer in zip(trained_layers, streamlit_layers):
        if trained_layer.get_weights():
            streamlit_layer.set_weights(trained_layer.get_weights())
    
    return streamlit_model

def main():
    """Main training function"""
    print("Starting Enhanced MNIST Model Training...")
    print("=" * 50)
    
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data()
    
    # Create the enhanced model
    model = create_enhanced_model()
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train the model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load the best model
    best_model = keras.models.load_model('best_mnist_model.h5')
    
    # Evaluate the model
    print("\nEvaluating the best model:")
    accuracy = evaluate_model(best_model, X_val, y_val)
    
    # Create Streamlit-compatible model
    print("\nCreating Streamlit-compatible model...")
    streamlit_model = create_model_for_streamlit(best_model, X_train)
    
    # Save the Streamlit-compatible model
    streamlit_model.save('mnist_model.h5')
    print("Enhanced model saved as 'mnist_model.h5'")
    
    print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
    print("The enhanced model is ready for use with your Streamlit app!")

if __name__ == "__main__":
    main()
