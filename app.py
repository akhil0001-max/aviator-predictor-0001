import streamlit as st
import base64
from PIL import Image
import io
import vision_trainer  # OCR + Predictor module

st.title("Aviator AI Predictor via Camera Snapshot")

st.markdown("### Step 1: Allow camera access and take snapshot below")

# HTML + JS to access webcam
camera_html = """
    <div>
        <video id="video" width="100%" autoplay></video><br>
        <button id="snap">Capture</button>
        <canvas id="canvas" style="display:none;"></canvas>
        <script>
            const video = document.getElementById('video');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => video.srcObject = stream);

            const canvas = document.getElementById('canvas');
            const snap = document.getElementById('snap');

            snap.addEventListener('click', () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                const image = canvas.toDataURL('image/png');
                window.parent.postMessage({ image: image }, '*');
            });
        </script>
    </div>
"""

st.components.v1.html(camera_html, height=300)

# Capture image from frontend (via JS event)
image_data = st.experimental_get_query_params().get("image")
if image_data:
    st.image(image_data[0], caption="Captured Frame")
    image_bytes = base64.b64decode(image_data[0].split(",")[1])
    image = Image.open(io.BytesIO(image_bytes))

    st.markdown("### Step 2: AI Prediction")

    with st.spinner("Analyzing..."):
        result = vision_trainer.predict_from_image(image)
        st.success(f"Predicted Multiplier: {result}")
