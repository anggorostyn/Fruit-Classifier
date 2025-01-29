import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Set page config
st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="wide")

# Load model once
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model('model/fruit_classifier_model.h5')

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)

# Predict fruit
@st.cache_data()
def predict_fruit(image, model):
    class_names = ["freshapple", "freshbanana", "freshorange", "rottenapple", "rottenbanana", "rottenorange"]
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return class_names[np.argmax(predictions[0])], float(np.max(predictions[0]))

# Process image input
def process_image_input(uploaded_file, model):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify"):
            predicted_class, confidence = predict_fruit(image, model)
            st.success(f"Prediction: {predicted_class} ({confidence:.2%})")

# Process video input (single frame prediction)
def process_video_input(uploaded_file, model):
    if uploaded_file is not None:
        cap = cv2.VideoCapture(uploaded_file.name)
        ret, frame = cap.read()
        cap.release()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(image, caption="Extracted Frame", use_column_width=True)
            predicted_class, confidence = predict_fruit(image, model)
            st.success(f"Prediction: {predicted_class} ({confidence:.2%})")
        else:
            st.error("Failed to read video file.")

# Process camera input (single frame prediction)
def process_camera_input(model):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(image, caption="Captured Frame", use_column_width=True)
        predicted_class, confidence = predict_fruit(image, model)
        st.success(f"Prediction: {predicted_class} ({confidence:.2%})")
    else:
        st.error("Failed to access camera.")

# Main function
def main():
    st.title("🍎 Fruit Classifier")
    st.sidebar.title("Options")
    input_type = st.sidebar.radio("Choose Input Type", ["Image", "Video", "Camera"])
    
    model = load_model()
    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        process_image_input(uploaded_file, model)
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        process_video_input(uploaded_file, model)
    else:
        process_camera_input(model)

if __name__ == "__main__":
    main()
