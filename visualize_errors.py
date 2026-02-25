
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import CNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./saved_models/mnist_cnn.pth"

    # MNIST normalization used in training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    wrong = []  # will store (image_tensor, true_label, pred_label)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            mismatches = preds != labels
            if mismatches.any():
                wrong_imgs = images[mismatches].cpu()
                wrong_true = labels[mismatches].cpu()
                wrong_pred = preds[mismatches].cpu()

                for i in range(wrong_imgs.size(0)):
                    wrong.append((wrong_imgs[i], int(wrong_true[i]), int(wrong_pred[i])))

            if len(wrong) >= 36:  # collect enough for a 6x6 grid
                break

    print(f"Collected {len(wrong)} wrong predictions (showing up to 36).")

    # Plot
    n = min(36, len(wrong))
    cols = 6
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(12, 8))
    for i in range(n):
        img, y_true, y_pred = wrong[i]

        # undo normalization to display nicely:
        # x_norm = (x - mean)/std -> x = x_norm*std + mean
        img_disp = img * 0.3081 + 0.1307
        img_disp = img_disp.clamp(0, 1)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_disp.squeeze(0), cmap="gray")
        plt.title(f"T:{y_true} P:{y_pred}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
