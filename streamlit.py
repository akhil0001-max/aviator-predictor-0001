import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Aviator Predictor", layout="centered")
st.title("Aviator Multiplier Predictor (AI-Powered)")

uploaded_file = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Screenshot', use_column_width=True)

    # OCR with EasyOCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.array(image))
    text = " ".join([res[1] for res in result])
    matches = re.findall(r'\d+\.\d+', text)
    multipliers = [float(match) for match in matches if float(match) < 100]

    if len(multipliers) >= 20:
        st.success(f"✅ {len(multipliers)} multipliers detected")
        
        # Plot chart
        st.subheader("Last 20 Multiplier Chart")
        plt.plot(multipliers[-20:], marker='o')
        plt.xlabel("Bet Index")
        plt.ylabel("Multiplier")
        st.pyplot(plt)

        # Prepare data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(np.array(multipliers).reshape(-1, 1))

        X, y = [], []
        for i in range(5, len(data)):
            X.append(data[i-5:i])
            y.append(data[i])
        X, y = np.array(X), np.array(y)

        # LSTM model
        model = Sequential([
            LSTM(64, input_shape=(X.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=30, verbose=0)

        # Predict next 10
        predictions = []
        input_seq = data[-5:]
        for _ in range(10):
            pred = model.predict(input_seq.reshape(1, 5, 1), verbose=0)
            predictions.append(scaler.inverse_transform(pred)[0][0])
            input_seq = np.append(input_seq[1:], pred, axis=0)

        # Show predictions
        st.subheader("Predicted Next 10 Multipliers")
        for i, val in enumerate(predictions, 1):
            st.write(f"🔮 Bet {i}: {val:.2f}x")
    else:
        st.warning("At least 20 multipliers needed for prediction.")
else:
    st.info("Please upload a screenshot of Aviator history (showing multipliers).")
