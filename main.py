import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

# XOR data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Build and train
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, learning_rate=0.5)
nn.train(X, y, epochs=5000)

# Predict
predictions = nn.predict(X)
print("Predictions:")
print(predictions)

# Accuracy
acc = nn.accuracy(y, predictions)
print(f"Accuracy: {acc * 100:.2f}%")

# Plot Loss
plt.plot(nn.losses)
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
