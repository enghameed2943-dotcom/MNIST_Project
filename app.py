
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import CNN
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_probabilities(probs):
    fig, ax = plt.subplots()
    ax.bar(range(10), probs)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

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

import torch.nn.functional as F

def predict(image):
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, 1)

    return (
        pred.item(),
        confidence.item(),
        probs.cpu().numpy()[0]
    )

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

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

    # Invert
    gray = 255 - gray

    # Blur slightly (reduce harsh edges)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        digit = thresh[y:y+h, x:x+w]

        # Resize longest side to 20
        if h > w:
            new_h = 20
            new_w = int(w * (20 / h))
        else:
            new_w = 20
            new_h = int(h * (20 / w))

        digit = cv2.resize(digit, (new_w, new_h))

        # Create 28x28 canvas
        canvas28 = np.zeros((28, 28), dtype=np.uint8)

        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

        img = Image.fromarray(canvas28)

        if st.button("Predict Drawing"):
            pred, conf, probs = predict(img)
            st.success(f"Prediction: {pred}")
            st.info(f"Confidence: {conf*100:.2f}%")
            show_probabilities(probs)