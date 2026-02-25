import torch
from torchvision import transforms
from PIL import Image
from src.model import CNN

def load_model(model_path, device):
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(image_path, model_path="./saved_models/mnist_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = Image.open(image_path)
    x = transform(img).unsqueeze(0).to(device)  # (1,1,28,28)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return pred

if __name__ == "__main__":
    # Example usage:
    # print(predict_image("your_digit.png"))
    pass
