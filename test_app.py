import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import os

# Configure page
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

def main():
    """Fallback Streamlit application for testing deployment"""
    # Title and description
    st.title("🔢 MNIST Digit Classifier")
    st.markdown("### Draw a digit (0-9) - Deployment Test Version")
    
    st.error("⚠️ TensorFlow loading issue detected. This is a test deployment to verify Streamlit Cloud setup.")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎨 Draw Here")
        
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=15,
            stroke_color="rgba(0, 0, 0, 1)",
            background_color="rgba(255, 255, 255, 1)",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # Clear button
        if st.button("🗑️ Clear Canvas", type="secondary"):
            st.rerun()
    
    with col2:
        st.markdown("#### 🤖 Deployment Status")
        st.info("Canvas working! TensorFlow installation in progress...")
        
        if canvas_result.image_data is not None:
            st.success("✅ Drawing detected!")
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            img = img.convert('L').resize((28, 28))
            st.image(img, width=140, caption="Processed Image")
        else:
            st.info("👆 Draw to test canvas functionality!")

    # Model information
    st.markdown("---")
    st.markdown("### 🧠 Model Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Images", "42,000")
    with col2:
        st.metric("Test Images", "8,400")
    with col3:
        st.metric("Model Architecture", "Enhanced CNN")
    with col4:
        st.metric("Validation Accuracy", "99.43%")

if __name__ == "__main__":
    main()
