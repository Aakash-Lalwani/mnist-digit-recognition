import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to 0-1 range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data for neural network (flatten 28x28 to 784)
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    
    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Create a neural network model"""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Architecture:")
    model.summary()
    
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """Train the model with callbacks"""
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def evaluate_model(model, x_test, y_test):
    """Evaluate model performance"""
    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Get predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return test_accuracy

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    """Main training pipeline"""
    print("🚀 Starting MNIST Digit Classification Training")
    print("=" * 50)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    print("\n🏗️ Creating neural network model...")
    model = create_model()
    
    # Train model
    print("\n🎯 Training model...")
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    accuracy = evaluate_model(model, x_test, y_test)
    
    # Plot training history
    print("\n📊 Plotting training history...")
    plot_training_history(history)
    
    # Save the model
    model.save('mnist_model.h5')
    print(f"\n✅ Model saved as 'mnist_model.h5' with accuracy: {accuracy:.4f}")
    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd

# Configure page
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    """Load the pre-trained model (cached for performance)"""
    try:
        model = tf.keras.models.load_model('mnist_model.h5')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please run 'python train_model.py' first to train the model.")
        return None

def preprocess_canvas_image(canvas_result):
    """Preprocess the drawn image for prediction"""
    if canvas_result.image_data is not None:
        # Convert to PIL Image
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Invert colors (white background, black digit -> black background, white digit)
        img_array = 255 - img_array
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, 784)
        
        return img_array, img
    return None, None

def predict_digit(model, processed_image):
    """Make prediction on the processed image"""
    if processed_image is not None and model is not None:
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        return predicted_digit, confidence, prediction[0]
    return None, None, None

def main():
    """Main Streamlit application"""
    # Title and description
    st.title("🔢 MNIST Digit Classifier")
    st.markdown("### Draw a digit (0-9) and watch the AI predict it in real-time!")
    
    # Load model (only once due to caching)
    model = load_trained_model()
    
    if model is None:
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎨 Draw Here")
        
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",  # Black fill
            stroke_width=15,
            stroke_color="rgba(0, 0, 0, 1)",  # Black stroke
            background_color="rgba(255, 255, 255, 1)",  # White background
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # Clear button
        if st.button("🗑️ Clear Canvas", type="secondary"):
            st.rerun()
    
    with col2:
        st.markdown("#### 🤖 AI Prediction")
        
        # Process and predict
        if canvas_result.image_data is not None:
            processed_img, pil_img = preprocess_canvas_image(canvas_result)
            
            if processed_img is not None:
                predicted_digit, confidence, probabilities = predict_digit(model, processed_img)
                
                if predicted_digit is not None:
                    # Display prediction
                    st.markdown(f"### Predicted Digit: **{predicted_digit}**")
                    st.markdown(f"### Confidence: **{confidence:.1f}%**")
                    
                    # Show processed image
                    st.markdown("#### Processed Image (28x28)")
                    if pil_img:
                        st.image(pil_img, width=140)
                    
                    # Show probability distribution
                    st.markdown("#### Probability Distribution")
                    prob_data = {
                        'Digit': list(range(10)),
                        'Probability': [f"{p*100:.1f}%" for p in probabilities]
                    }
                    
                    # Create a bar chart
                    chart_data = {str(i): probabilities[i] for i in range(10)}
                    st.bar_chart(chart_data)
                    
                    # Display probability table
                    df = pd.DataFrame(prob_data)
                    st.dataframe(df, hide_index=True)
        
        else:
            st.info("👆 Draw a digit on the canvas to see the prediction!")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. **Draw**: Use your mouse or touch to draw a digit (0-9) on the canvas
    2. **Predict**: The AI will automatically predict your digit in real-time
    3. **Clear**: Click the 'Clear Canvas' button to start over
    4. **Tips**: 
       - Draw clearly and make the digit fill most of the canvas
       - Use thick strokes for better recognition
       - Try different writing styles to test the model's robustness
    """)
    
    # Model information
    st.markdown("---")
    st.markdown("### 🧠 Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Images", "60,000")
    with col2:
        st.metric("Test Images", "10,000")
    with col3:
        st.metric("Model Architecture", "Dense Neural Network")
    
    st.markdown("""
    **Model Details:**
    - **Architecture**: Multi-layer Neural Network with Dropout
    - **Layers**: 784 → 512 → 256 → 128 → 10 neurons
    - **Activation**: ReLU (hidden), Softmax (output)
    - **Dataset**: MNIST (28×28 grayscale digit images)
    - **Accuracy**: ~98%+ on test data
    """)

if __name__ == "__main__":
    main()
