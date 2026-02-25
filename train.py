import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import CNN

def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    # ========== Config ==========
    batch_size = 64
    lr = 1e-3
    epochs = 5
    data_dir = "./data"
    save_path = "./saved_models/mnist_cnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ========== Transforms ==========
    # ToTensor converts (0..255) -> float (0..1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # common MNIST mean/std
    ])

    # ========== Dataset ==========
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ========== Model ==========
    model = CNN().to(device)

    # ========== Loss + Optimizer ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ========== Training ==========
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_acc = accuracy(model, train_loader, device)
        test_acc  = accuracy(model, test_loader, device)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to: {save_path} (Test Acc: {best_acc:.4f})")

    print("Training done. Best Test Acc:", best_acc)

if __name__ == "__main__":
    main()
