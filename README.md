# 🧠 Neural Network from Scratch (XOR Problem)

This project is a minimal neural network built **from scratch using only NumPy**, designed to solve the classic XOR problem. It demonstrates how forward propagation, backpropagation, and gradient descent work at a low level — without using high-level libraries like TensorFlow or PyTorch.

> ✅ Great for learning the fundamentals of deep learning by building it yourself.

---

## 🔍 Problem: XOR Gate

The XOR problem is a simple binary classification task that cannot be solved by a linear model. This neural network learns to output the correct XOR logic:

| Input | Output |
|-------|--------|
| [0, 0] |   0    |
| [0, 1] |   1    |
| [1, 0] |   1    |
| [1, 1] |   0    |

---

## 📌 Key Features

- ✅ Feedforward neural network (fully connected)
- 🧮 Manual implementation of:
  - Forward pass
  - Backpropagation
  - Binary cross-entropy loss
- 🔁 Trained using gradient descent
- 📈 Loss tracking and visualization
- 📊 Accuracy evaluation
- 🧠 Sigmoid activation functions

---

## 🧪 Example Output

Epoch 4900: Loss = 0.0153
Epoch 4999: Loss = 0.0148
Predictions:
[[0]
[1]
[1]
[0]]
Accuracy: 100.00%

<p align="center">
  <img src="https://raw.githubusercontent.com/Isma3ilovitch/neural-network-from-scratch/main/assets/loss_plot.png" alt="Loss Plot" width="500"/>
</p>

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Isma3ilovitch/neural-network-from-scratch.git
cd neural-network-from-scratch
```
---
## 💡 Learning Goal
This project is educational. It’s ideal for understanding the inner workings of neural networks without relying on black-box libraries.
---
## 📜 License
MIT License — feel free to use, modify, and share!
---
## 🙌 Acknowledgments
Inspired by Andrew Ng’s Deep Learning course, and CS50AI XOR example.





