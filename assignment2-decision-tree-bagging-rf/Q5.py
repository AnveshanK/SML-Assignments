import numpy as np
import matplotlib.pyplot as plt

# Generate Data
# np.random.seed(116)  
# x = np.linspace(0, 2 * np.pi, 100)
x = np.random.uniform(0, 2 * np.pi, 100)
x = np.sort(x)
# print(x)

y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

def Calculate_W(x_train, y_train, degree):
    X = np.column_stack([x_train**i for i in range(degree, -1, -1)])
    W = np.linalg.pinv(X.T @ X) @ X.T @ y_train  
    return W

# Perform 5-Fold Cross-Validation
def cross_validation(x, y, degree, k=5):
    indices = np.arange(len(x))
    x, y = x[indices], y[indices]

    fold_size = len(x) // k
    val_error = []
    train_error = []
    
    for i in range(k):
        if i == k - 1:  
            val_indices = indices[i * fold_size:]  
        else:
            val_indices = indices[i * fold_size:(i + 1) * fold_size]

        train_indices = np.setdiff1d(indices, val_indices)
        
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]
        
        W = Calculate_W(x_train, y_train, degree)
        
        X_train = np.column_stack([x_train**i for i in range(degree, -1, -1)])
        Y_train_pred = X_train @ W

        l2_norm_train = np.sum((y_train - Y_train_pred)**2)
        train_error.append(l2_norm_train)

        X_val = np.column_stack([x_val**i for i in range(degree, -1, -1)])
        y_pred = X_val @ W
        
        l2_norm_val = np.sum((y_val - y_pred) ** 2)
        val_error.append(l2_norm_val)
    
    return np.mean(val_error)

# Consider models upto degree 4
degrees = [1, 2, 3, 4]
best_degree = None
min_val_error = float('inf')  

for d in degrees:
    avg_val_error = cross_validation(x, y, d)  
    if avg_val_error < min_val_error:
        min_val_error = avg_val_error
        best_degree = d

# Visualization
W = Calculate_W(x, y, best_degree)
x_train = x
y_train = y
X_train = np.column_stack([x_train**i for i in range(best_degree, -1, -1)])
y_pred = X_train @ W

plt.figure(figsize=(16, 9))

# true function
plt.plot(x_train, np.sin(x_train), label="True Function ( y=sin(x) )", color='black')

# noisy training points
plt.scatter(x_train, y_train, color="blue", label="Noisy Training Points",s=10)

# regression model’s prediction
plt.plot(x_train, y_pred, label=f"Regeression Model Prediction (Degree : {best_degree})", color='red',linestyle="dashed")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Regression model using 5-fold Cross-Validation")
plt.show()

print("Regression Model Prediction")
print(f"Best polynomial degree : {best_degree}")