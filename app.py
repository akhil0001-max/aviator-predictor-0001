import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import pytesseract

# Load trained model (make sure 'your_model.h5' is present)
model = load_model("your_model.h5")

# Title
st.title("Aviator AI Predictor")
st.write("Upload a screenshot from the Aviator game to get predictions.")

# Upload section
uploaded_file = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    # Optional: OCR to check input
    text = pytesseract.image_to_string(image)
    st.write("Extracted Text (Optional):")
    st.code(text)

    # Resize + normalize image
    resized = image.resize((224, 224))
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    confidence = float(np.max(prediction)) * 100
    predicted_class = np.argmax(prediction)

    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
