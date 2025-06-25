import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Generate synthetic data (2 x 10 for each class)
np.random.seed(2023116)
n_samples = 10

# Class 0: mean = [-1, -1], covariance = identity
X0 = np.random.multivariate_normal(mean=[-1, -1], cov=np.eye(2), size=n_samples).T  # shape (2, 10)
y0 = np.zeros((1, n_samples))  # shape (1, 10)

# Class 1: mean = [1, 1], covariance = identity
X1 = np.random.multivariate_normal(mean=[1, 1], cov=np.eye(2), size=n_samples).T  # shape (2, 10)
y1 = np.ones((1, n_samples))  # shape (1, 10)

# Combine data: (2, 20) and labels (1, 20)
X = np.hstack((X0, X1))  # shape (2, 20)
y = np.hstack((y0, y1))  # shape (1, 20)

# Shuffle columns (samples)
perm = np.random.permutation(X.shape[1])
X = X[:, perm]
y = y[:, perm]

# Train-test split (50-50)
X_train, X_test = X[:, :10], X[:, 10:]
y_train, y_test = y[:, :10], y[:, 10:]

# Network parameters
input_dim = 2
hidden_dim = 1
output_dim = 1
lr = 0.1
rounds = 1000

# Initialize weights/biases
W1 = np.random.randn(hidden_dim, input_dim)  # shape (1, 2)
b1 = np.random.randn(hidden_dim, 1)          # shape (1, 1)
W2 = np.random.randn(output_dim, hidden_dim) # shape (1, 1)
b2 = np.random.randn(output_dim, 1)          # shape (1, 1)

# Training loop
for round in range(rounds):
    # Forward pass
    Z1 = W1 @ X_train + b1          # shape (1, 10)     (beta1 x + beta0)
    A1 = sigmoid(Z1)                # shape (1, 10)
    Z2 = W2 @ A1 + b2               # shape (1, 10)
    Y_pred = Z2                     # linear output

    # Compute MSE
    loss = np.mean((y_train - Y_pred) ** 2)

    # gradient descent
    dZ2 = 2 * (Y_pred - y_train) / y_train.shape[1]  # shape (1, 10)
    dW2 = dZ2 @ A1.T                                 # (1, 10) x (10, 1) = (1, 1)
    db2 = np.sum(dZ2, axis=1, keepdims=True)         # (1, 1)

    dA1 = W2.T @ dZ2                                 # (1, 1) x (1, 10) = (1, 10)
    dZ1 = dA1 * sigmoid_derivative(Z1)               # (1, 10)
    dW1 = dZ1 @ X_train.T                            # (1, 10) x (10, 2) = (1, 2)
    db1 = np.sum(dZ1, axis=1, keepdims=True)         # (1, 1)

    # Update weights
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1


# Predict on test set
Z1_test = W1 @ X_test + b1
A1_test = sigmoid(Z1_test)
Z2_test = W2 @ A1_test + b2
Y_test_pred = Z2_test

mse_test = np.mean((y_test - Y_test_pred) ** 2)
print("Test MSE:", mse_test)

