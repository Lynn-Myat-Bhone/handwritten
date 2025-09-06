import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom


# -----------------------------
# Load CNN Model
# -----------------------------
model = load_model("handwritten_cnn.keras")

st.set_page_config(page_title=" Digit Recognizer", layout="centered")
st.title("Digit Recognizer ")

# -----------------------------
# Preprocess Canvas Drawing
# -----------------------------
def process_image(image_data, size=28):
    """Convert drawn image to grayscale and resize to 28x28 for CNN"""
    from scipy.ndimage import zoom
    import numpy as np
    
    # Convert to grayscale
    grayscale_image = np.sum(image_data, axis=2)  # shape (H,W)
    
    # Resize proportionally
    zoom_factor = size / grayscale_image.shape[0]
    resized_image = zoom(grayscale_image, zoom_factor)
    
    # Normalize pixels
    normalized_image = resized_image.astype(np.float32) / 255.0
    
    # Reshape to (1,28,28,1) for CNN
    return normalized_image.reshape(1, size, size, 1)



canvas_result = st_canvas(
    stroke_width=10, height=28*5, width=28*5,
    drawing_mode="freedraw", key="canvas", update_streamlit=True
)

if canvas_result.image_data is not None and np.any(canvas_result.image_data):
    processed_image = process_image(canvas_result.image_data)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    
    st.header("Prediction:")
    st.markdown(f"This number appears to be **:red[{predicted_digit}]**")
else:
    st.header("Prediction:")
    st.write("No number drawn, please draw a digit to get a prediction.")
