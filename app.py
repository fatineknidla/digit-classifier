import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
from PIL import Image
import io
import os

# Priority: Streamlit secrets > env var > local default
if "BACKEND_URL" in st.secrets:
    BACKEND_URL = st.secrets["BACKEND_URL"]
else:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Ensure the URL ends with /predict
if not BACKEND_URL.endswith("/predict"):
    BACKEND_URL = f"{BACKEND_URL.rstrip('/')}/predict"

st.title("MNIST Digit Recognizer")
st.write("Draw a digit (0-9) below!")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert canvas drawing to Image
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    img = img.convert('L') # Convert to Grayscale
    
    if st.button("Predict"):
        # Send to FastAPI
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        try:
            files = {"file": ("image.png", byte_im, "image/png")}
            response = requests.post(BACKEND_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.header(f"Result: {data['prediction']}")
                st.subheader(f"Confidence: {data['confidence']}")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")