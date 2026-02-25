
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import CNN
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

st.title("🧠 MNIST Handwritten Digit Recognizer")
st.markdown("### Upload or Draw a Digit (0–9)")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load("./saved_models/mnist_cnn.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict(image):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return pred

# -----------------------------
# Upload Section
# -----------------------------
st.subheader("📤 Upload Image")

uploaded = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=200)

    pred = predict(img)
    st.success(f"Predicted Digit: {pred}")

# -----------------------------
# Draw Section
# -----------------------------
st.subheader("✏ Draw Digit")

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:

    img_array = canvas_result.image_data.astype(np.uint8)

    # Convert RGBA → grayscale
    img = Image.fromarray(img_array).convert("L")

    # Invert to match MNIST (white digit on black)
    img = Image.fromarray(255 - np.array(img))

    if st.button("Predict Drawing"):
        pred, conf, probs = predict(img)
        st.success(f"Prediction: {pred}")
        st.info(f"Confidence: {conf*100:.2f}%")
        show_probabilities(probs)