from PIL import Image
import pytesseract
import re
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("your_model.h5")  # Pre-trained LSTM or other model

def predict_from_image(image: Image.Image) -> str:
    gray_image = image.convert("L")
    text = pytesseract.image_to_string(gray_image)
    print("Extracted:", text)

    # Extract multipliers from OCR text
    multipliers = re.findall(r"\d+\.\d+", text)
    data = [float(m) for m in multipliers if float(m) < 100]

    if len(data) < 10:
        return "Insufficient data"

    input_data = np.array(data[-10:]).reshape(1, 10, 1)
    prediction = model.predict(input_data)
    return str(round(prediction[0][0], 2))
