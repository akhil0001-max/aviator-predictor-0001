
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import re

# Function to create dataset
def create_dataset(series, seq_length):
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series[i:i + seq_length])
        y.append(series[i + seq_length])
    return np.array(X), np.array(y)

# Upload initial CSV or input list
print("Please upload CSV file with multiplier values (1 column, no header)")
uploaded = files.upload()
file_name = next(iter(uploaded))
data = pd.read_csv(file_name, header=None).iloc[:,0].tolist()

# Prepare dataset
seq_len = 10
X, y = create_dataset(data, seq_len)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build model
model = Sequential([
    LSTM(64, input_shape=(seq_len, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, verbose=1)

# Predict next 10
input_seq = np.array(data[-seq_len:]).reshape(1, seq_len, 1)
predictions = []
for _ in range(10):
    pred = model.predict(input_seq, verbose=0)[0][0]
    predictions.append(round(pred, 2))
    input_seq = np.append(input_seq[:,1:,:], [[[pred]]], axis=1)

# Real-time feedback loop
for pred in predictions:
    print(f"Prediction: {pred}")
    feedback = input("Kya prediction sahi tha? (yes/no): ").strip().lower()

    if feedback == 'no':
        method = input("Screenshot se lena hai? (y/n): ").strip().lower()
        if method == 'y':
            uploaded = files.upload()
            img = Image.open(next(iter(uploaded)))
            text = pytesseract.image_to_string(img)
            found = re.findall(r"\d+\.\d+x", text)
            if found:
                actual = float(found[0][:-1])
            else:
                actual = float(input("OCR fail. Manually daalo: "))
        else:
            actual = float(input("Actual value: "))
        
        data.append(actual)
        X, y = create_dataset(data, seq_len)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model.fit(X, y, epochs=10, verbose=0)
    
    time.sleep(3)
