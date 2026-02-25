# 🧠 MNIST Handwritten Digit Recognition (PyTorch + Streamlit)

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits (0–9) using the MNIST dataset.

The system includes:

- End-to-end training pipeline
- Model evaluation (confusion matrix + error analysis)
- Visualization of misclassified samples
- Web deployment with Streamlit
- Real-time digit prediction (Upload + Drawing Canvas)

---

## 🎯 Problem Definition

Handwritten Digit Recognition is a multi-class image classification task where:

- Input: 28×28 grayscale image
- Output: One digit from 0–9
- Objective: Minimize classification error using deep learning

---

## 🏗 Model Architecture

- Conv2D (32 filters) + BatchNorm + ReLU
- Conv2D (64 filters) + BatchNorm + ReLU
- MaxPooling
- Dropout Regularization
- Fully Connected Layer
- CrossEntropy Loss
- Adam Optimizer

---

## 📊 Model Performance

- Training Accuracy: ~99.5%
- Test Accuracy: **99.1%**
- Minimal overfitting
- Strong generalization

---

## 🖥 Web Application Features

- 📤 Upload digit image
- ✏ Draw digit directly in browser
- 📊 Confidence probability visualization
- ⚡ Real-time inference

---

## 🛠 Tech Stack

- Python
- PyTorch
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
MNIST_Project/
│
├── src/
│   └── model.py
│
├── train.py
├── evaluate.py
├── visualize_errors.py
├── infer.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 📌 Future Improvements

- ONNX export
- Mobile deployment (Android)
- Cloud API version (FastAPI)

---

## 👨‍💻 Author

Hameed  
Deep Learning & AI Engineering Portfolio Project
