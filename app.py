import streamlit as st
from PIL import Image
import numpy as np
import pytesseract
import re
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("your_model.h5")  # Replace with your actual model file

st.title("Aviator AI Predictor")
st.markdown("Upload screenshot and get next round multiplier prediction")

uploaded_file = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    # OCR
    text = pytesseract.image_to_string(image)
    st.markdown("**Extracted Text:**")
    st.code(text)

    # Extract multipliers
    matches = re.findall(r'\d+\.\d+', text)
    multipliers = [float(m) for m in matches if float(m) < 100]

    if len(multipliers) < 10:
        st.warning("Need at least 10 multipliers for prediction. Found: " + str(len(multipliers)))
    else:
        input_data = np.array(multipliers[-10:]).reshape(1, 10, 1)
        prediction = model.predict(input_data)
        predicted_value = round(prediction[0][0], 2)
        
        confidence = round(float(abs(predicted_value - np.mean(multipliers)) / np.std(multipliers)) * 100, 2)
        confidence = min(confidence, 99.9)  # Limit max confidence to 99.9%

        st.success(f"Predicted Next Multiplier: **{predicted_value}x**")
        st.info(f"Confidence Level: **{confidence}%**")
