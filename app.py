
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import CNN
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
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




import numpy as np
from PIL import Image, ImageOps

def preprocess_canvas(image_rgba: np.ndarray) -> Image.Image:
    # RGBA -> grayscale
    img = Image.fromarray(image_rgba.astype(np.uint8)).convert("L")

    # Make sure background is black, digit is white (canvas is usually already this)
    # If your canvas background is black and stroke is white, you usually DON'T invert.
    # But some setups still need it. We'll auto-decide based on mean brightness:
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)

    arr = np.array(img)

    # Find where the digit pixels are (simple threshold, not heavy binarization)
    ys, xs = np.where(arr > 20)
    if len(xs) == 0 or len(ys) == 0:
        return img.resize((28, 28))

    # Crop bounding box
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))

    # Resize keeping aspect ratio so max side = 20
    w, h = cropped.size
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Paste into 28x28 center
    canvas28 = Image.new("L", (28, 28), 0)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas28.paste(resized, (x_off, y_off))

    return canvas28
# -----------------------------
# Process Drawing
# -----------------------------
if canvas_result.image_data is not None:

    img = preprocess_canvas(canvas_result.image_data)

    # Show what model actually sees
    st.image(img.resize((140, 140)), caption="Model input (28x28 scaled up)")

    if st.button("Predict Drawing"):
        pred, conf, probs = predict(img)
        st.success(f"Prediction: {pred}")
        st.info(f"Confidence: {conf*100:.2f}%")
        show_probabilities(probs)