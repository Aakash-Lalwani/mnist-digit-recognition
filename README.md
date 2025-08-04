# 🎨 Enhanced MNIST Digit Recognition App

An interactive web application for handwritten digit recognition using an enhanced CNN model with **99.43% accuracy**.

## ✨ Features

- **Interactive Drawing Canvas**: Draw digits with your mouse
- **Real-time Prediction**: Instant digit recognition as you draw
- **Enhanced CNN Model**: Achieves 99.43% validation accuracy
- **Data Augmentation**: Robust model with rotation, zoom, and translation
- **Advanced Architecture**: BatchNormalization, Dropout, and optimized layers

## 🚀 Live Demo

Try the app live: [Your App URL will be here after deployment]

## 🎯 Model Performance

- **Validation Accuracy**: 99.43%
- **Architecture**: Enhanced CNN with 3 convolutional blocks
- **Training**: 42,000 MNIST samples with data augmentation
- **Optimization**: Adam optimizer with learning rate scheduling

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: TensorFlow/Keras
- **Model**: Enhanced Convolutional Neural Network
- **Deployment**: Streamlit Community Cloud

## 📋 Local Setup

```bash
# Clone the repository
git clone [your-repo-url]
cd MNIST

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run MNISTproj.py
```

## 🎨 How to Use

1. **Draw**: Use your mouse to draw a digit (0-9) on the canvas
2. **Predict**: The model will instantly recognize your digit
3. **Clear**: Use the clear button to draw a new digit
4. **Experiment**: Try different writing styles and see the robust predictions

## 📊 Model Architecture

- **Data Augmentation**: Random rotation, zoom, translation
- **Conv2D Layers**: 32, 64, 128 filters with ReLU activation
- **Regularization**: BatchNormalization and Dropout layers
- **Dense Layers**: 512 and 256 neurons with regularization
- **Output**: 10-class softmax for digit classification

## 🏆 Performance Highlights

- **99.43% Validation Accuracy** - Exceptional performance
- **Real-time Inference** - Fast predictions for interactive use
- **Robust Recognition** - Handles various drawing styles
- **Production Ready** - Optimized for deployment

---

Built with ❤️ using TensorFlow and Streamlit
