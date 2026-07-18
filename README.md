# 🧠 Handwritten Digit Recognizer using Deep Learning (MNIST)

A Deep Learning project that recognizes handwritten digits (0–9) using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The model is trained on the MNIST dataset and achieves high accuracy in digit classification.

---

## 📌 Project Overview

Handwritten digit recognition is one of the most popular introductory problems in Deep Learning and Computer Vision. This project uses a fully connected Artificial Neural Network (ANN) to classify handwritten digits from grayscale images.

The model is trained on the MNIST dataset containing 70,000 images of handwritten digits and learns to identify patterns in pixel values to accurately predict the corresponding digit.

---

## 🚀 Features

- Handwritten digit classification (0–9)
- Deep Learning model built with TensorFlow/Keras
- Data preprocessing and normalization
- Training and validation visualization
- Model saving for future predictions
- Achieves approximately **98.25% test accuracy**

---

<img width="887" height="523" alt="image" src="https://github.com/user-attachments/assets/e35a031d-e895-4497-bade-8c8e304f5379" />

## 🛠️ Technologies Used

- Python 3
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## 📂 Dataset

### MNIST Dataset

The MNIST dataset consists of:

| Dataset | Images |
|----------|---------|
| Training Set | 60,000 |
| Test Set | 10,000 |

Each image:
- Size: 28 × 28 pixels
- Grayscale
- Represents digits from 0–9

---

## 🏗️ Model Architecture

```text
Input Layer (784 neurons)
        ↓
Dense Layer (128 neurons, ReLU)
        ↓
Dense Layer (64 neurons, ReLU)
        ↓
Output Layer (10 neurons, Softmax)
```

### Optimizer
- Adam

### Loss Function
- Sparse Categorical Crossentropy

### Evaluation Metric
- Accuracy

---

## 📊 Training Results

| Metric | Value |
|----------|----------|
| Epochs | 15 |
| Training Samples | 60,000 |
| Test Samples | 10,000 |
| Test Loss | 0.0682 |
| Test Accuracy | 98.25% |

---

## 📈 Accuracy & Loss Curves

### Training Accuracy

<img width="532" height="343" alt="image" src="https://github.com/user-attachments/assets/1e6abb97-331e-4b2c-9546-8631b25bc156" />


### Training Loss

<img width="525" height="343" alt="image" src="https://github.com/user-attachments/assets/54e1e930-17fd-4656-9a0d-9dc06068dbbd" />


> Replace the image paths above with your actual screenshots after uploading them to GitHub.

---

## 🔍 Sample Predictions



| Actual Digit | Predicted Digit |
|--------------|------------------|
| 7 | 7 |
| 3 | 3 |
| 0 | 0 |
| 5 | 5 |
| 8 | 8 |

Example prediction output: 
<img width="1090" height="558" alt="image" src="https://github.com/user-attachments/assets/611d1899-58d6-4734-8bca-ce8daebb8e31" />


```text
Actual: 7
Predicted: 7
Confidence: 99.8%
```

---

## 📁 Project Structure

```text
MNIST-Handwritten-Digit-Recognizer/
│
├── mnist_recognizer.py
├── mnist_ann_model.keras
├── README.md
├── requirements.txt
│
├── images/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── sample_predictions.png
│
└── outputs/
    └── prediction_results.txt
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/MNIST-Handwritten-Digit-Recognizer.git
```

### 2. Navigate to Project

```bash
cd MNIST-Handwritten-Digit-Recognizer
```

### 3. Create Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

#### Linux / Ubuntu

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install tensorflow numpy matplotlib
```

Or

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Execute:

```bash
python mnist_recognizer.py
```

The model will:

1. Load the MNIST dataset
2. Preprocess the data
3. Train the neural network
4. Evaluate performance
5. Save the trained model
6. Display accuracy and loss graphs

---

## 💾 Saved Model

After training, the model is saved as:

```text
mnist_ann_model.keras
```

Load it later using:

```python
from tensorflow.keras.models import load_model

model = load_model("mnist_ann_model.keras")
```

---

## 📚 Learning Outcomes

Through this project, I learned:

- Deep Learning fundamentals
- Artificial Neural Networks (ANN)
- TensorFlow and Keras workflows
- Data preprocessing techniques
- Model training and evaluation
- Accuracy and loss visualization
- Saving and loading trained models

---

## 🔮 Future Improvements

- Implement Convolutional Neural Networks (CNNs)
- Build a GUI for digit drawing and prediction
- Deploy as a web application
- Improve accuracy using data augmentation
- Real-time digit recognition using webcam input

---

## 👩‍💻 Author

**Divyanshi Yadav**

B.Tech CSE (AI & Deep Learning)  
Mody University

GitHub: https://github.com/divya247git

---

## ⭐ Results Summary

✅ Deep Learning ANN Model  
✅ Trained on MNIST Dataset  
✅ 98.25% Test Accuracy  
✅ TensorFlow/Keras Implementation  
✅ Model Saved for Future Predictions
