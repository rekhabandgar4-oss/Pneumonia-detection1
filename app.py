import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")

st.title("🫁 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to detect pneumonia")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/saved_models/pneumonia_detection_model.h5')

model = load_model()

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", width=300)
    
    # Convert to RGB and resize
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            prediction = model.predict(img)[0][0]
        
        if prediction > 0.5:
            st.error(f"**Prediction: PNEUMONIA**")
            st.warning(f"Confidence: {prediction*100:.2f}%")
        else:
            st.success(f"**Prediction: NORMAL**")
            st.info(f"Confidence: {(1-prediction)*100:.2f}%")
        
        st.write("### Probabilities:")
        st.write(f"Normal: {(1-prediction)*100:.2f}%")
        st.write(f"Pneumonia: {prediction*100:.2f}%")
