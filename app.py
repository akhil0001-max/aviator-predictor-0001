import streamlit as st
import vision_trainer
import easyocr
import numpy as np
from PIL import Image
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import time

st.set_page_config(page_title="Aviator Predictor", layout="centered")

# HTML + CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
        background-color: #0f0f0f;
        color: #f1f1f1;
    }

    .main-title {
        text-align: center;
        font-size: 3em;
        background: linear-gradient(90deg, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }

    .upload-button {
        background-color: #121212;
        color: #00f260;
        padding: 10px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        border: 2px solid #00f260;
        margin-top: 20px;
    }

    .prediction-box {
        background: #1e1e1e;
        padding: 12px;
        margin: 5px 0;
        border-radius: 8px;
        border-left: 6px solid #00f260;
        box-shadow: 0 0 12px rgba(0,255,100,0.2);
    }

    .footer {
        margin-top: 40px;
        text-align: center;
        font-size: 0.85em;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Aviator Predictor</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with st.spinner('Reading multipliers...'):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Screenshot", use_column_width=True)

        reader = easyocr.Reader(['en'])
        result = reader.readtext(np.array(image))
        text = " ".join([res[1] for res in result])
        matches = re.findall(r'\d+\.\d+', text)
        multipliers = [float(match) for match in matches if float(match) < 100]

        if len(multipliers) >= 20:
            st.success(f"{len(multipliers)} multipliers detected.")
            st.subheader("Last 20 Multipliers Chart")
            plt.plot(multipliers[-20:], marker='o', color='lime')
            st.pyplot(plt)

            scaler = MinMaxScaler()
            data = scaler.fit_transform(np.array(multipliers).reshape(-1, 1))

            X, y = [], []
            for i in range(5, len(data)):
                X.append(data[i-5:i])
                y.append(data[i])
            X, y = np.array(X), np.array(y)

            model = Sequential([
                LSTM(64, input_shape=(X.shape[1], 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=25, verbose=0)

            st.subheader("Next 10 Predictions")

            predictions = []
            input_seq = data[-5:]
            for _ in range(10):
                pred = model.predict(input_seq.reshape(1, 5, 1), verbose=0)
                predictions.append(scaler.inverse_transform(pred)[0][0])
                input_seq = np.append(input_seq[1:], pred, axis=0)

            for i, val in enumerate(predictions, 1):
                color = "#00f260" if val > 2 else "#f12711"
                st.markdown(
                    f"<div class='prediction-box' style='border-left-color:{color}'><b>Prediction {i}:</b> {val:.2f}x</div>",
                    unsafe_allow_html=True
                )

        else:
            st.warning("At least 20 multipliers required.")
else:
    st.markdown('<div class="upload-button">Upload a screenshot to begin prediction</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Made with Love by Akhil • AI Powered • 2025</div>', unsafe_allow_html=True)
