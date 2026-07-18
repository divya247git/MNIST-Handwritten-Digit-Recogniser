"""
Handwritten Digit Recognizer (MNIST) using a simple ANN (Dense layers)
Tech: TensorFlow / Keras
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
# Use Agg backend so matplotlib doesn't try to open GUI windows in a headless environment
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load and preprocess data
# ----------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images into 784-length vectors (ANN needs 1D input)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

# ----------------------------
# 2. Build the ANN model
# ----------------------------
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")  # 10 classes (digits 0-9)
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# 3. Train the model
# ----------------------------
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# ----------------------------
# 4. Evaluate on test set
# ----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# ----------------------------
# 5. Save the trained model
# ----------------------------
model.save("mnist_ann_model.keras")
print("Model saved as mnist_ann_model.keras")

# ----------------------------
# 6. Plot training history
# ----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.close()

# ----------------------------
# 7. Predict on a few test samples
# ----------------------------
predictions = model.predict(x_test[:10])
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample predictions vs actual:")
for i in range(10):
    print(f"Predicted: {predicted_labels[i]}, Actual: {y_test[i]}")

# ----------------------------
# 8. Visualize a few predictions
# ----------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = x_test[i].reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Pred: {predicted_labels[i]} / True: {y_test[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.close()
