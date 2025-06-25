import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import os

# Change the working directory to the script's location (required for data set, if only file and not the complete directory is opened in IDE)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)  # Skip the header
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(-1, 28 * 28).astype(np.float32) 


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)  # Skip the header (first 8 bytes)
        labels = np.frombuffer(f.read(), dtype=np.uint8)  
        return labels 

# Load dataset
train_images = load_mnist_images("TrainSet/train-images.idx3-ubyte")
train_labels = load_mnist_labels("TrainSet/train-labels.idx1-ubyte")

test_images = load_mnist_images("TestSet/t10k-images.idx3-ubyte")
test_labels = load_mnist_labels("TestSet/t10k-labels.idx1-ubyte")

# Filter only classes 0, 1, and 2
def filter_set(images,labels):
    filtered_images = []
    filtered_labels = []

    for i in range(len(labels)):
        if labels[i] in [0, 1, 2]:
            filtered_images.append(images[i])
            filtered_labels.append(labels[i])

    filtered_images = np.array(filtered_images)
    filtered_labels = np.array(filtered_labels)
    return filtered_images,filtered_labels

# Filter Sets
filtered_train_images,filtered_train_labels=filter_set(train_images,train_labels)
filtered_test_images,filtered_test_labels=filter_set(test_images,test_labels)

# Randomly sample 100 training and 100 test samples for class each 
def random_samples(images, labels):
    selected_images = []
    selected_labels = []
    for digit in [0, 1, 2]:
        indices = np.where(labels == digit)[0]
        selected_indices = np.random.choice(indices, 100, replace=False)
        selected_images.append(images[selected_indices])
        selected_labels.append(labels[selected_indices])
    return np.vstack(selected_images).T, np.hstack(selected_labels)

Train_Set_Images, Train_Set_Labels=random_samples(filtered_train_images,filtered_train_labels)
Test_Set_Images, Test_Set_Labels=random_samples(filtered_test_images,filtered_test_labels)

# Normalize pixel values to [0, 1]
Train_Set_Images = Train_Set_Images / 255.0    # this is final 784x300 data matrix where every 100 samples belong to different classes 0,1,2
Test_Set_Images =Test_Set_Images / 255.0

# dxn = 784x100 matrix for class each 
X_train_class_equals_0=Train_Set_Images[:,:100]       # class 0
X_train_class_equals_1=Train_Set_Images[:,100:200]    # class 1
X_train_class_equals_2=Train_Set_Images[:,200:300]    # class 2

train_set_class_matrices=[X_train_class_equals_0,X_train_class_equals_1,X_train_class_equals_2]

# Maximum Likelihood Estimation
def compute_mle(class_matrix):
    N = class_matrix.shape[1]  
    
    mu_c = np.sum(class_matrix, axis=1, keepdims=True) / N

    x_minus_mu = class_matrix - mu_c  
    sigma_c = np.dot(x_minus_mu,x_minus_mu.T) / N   # MLE gives biased estimate hence (N) is used and not (N-1)  

    return mu_c, sigma_c


# Compute MLE estimates for train set 
class_means=[]  # 3 elements 784x1 vector each 
class_covariances=[]    # 3 elements 784x784 matrix each 

# for Train set only
for matrix in train_set_class_matrices:   
    class_mean,class_cov=compute_mle(matrix)
    class_means.append(class_mean)
    class_covariances.append(class_cov)
train_set_global_mean=np.mean(Train_Set_Images,axis=1,keepdims=True)

# PCA implementation
def compute_PCA(X, variance_retained=0.95,n_components=-1):
    total_samples=X.shape[1]    # N=300
    Xc=X-train_set_global_mean
    covariance_S=np.dot(Xc,Xc.T)/(total_samples-1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_S)  

    sorted_indices = np.argsort(eigenvalues)[::-1]  
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    total_variance=np.sum(eigenvalues)
    required_variance=total_variance*variance_retained
    num_eigenvalues=len(eigenvalues)
    psum=0
    index=0
    while(index<num_eigenvalues and psum<required_variance):
        psum+=eigenvalues[index]
        index+=1
    p=index

    if n_components!=-1:
        p=n_components        

    U_p = eigenvectors[:, :p]   
    return U_p,p

# FDA implementation
def compute_FDA(X, class_matrices, class_means, global_mean):
    num_classes = len(class_matrices)  
    N_c = [x.shape[1] for x in class_matrices]  

    S_B = np.zeros((X.shape[0], X.shape[0]))  # (784x784)
    for c in range(num_classes):
        mean_diff = class_means[c] - global_mean
        S_B += N_c[c] * np.dot(mean_diff, mean_diff.T)

    S_W = np.zeros((X.shape[0], X.shape[0]))  # (784x784)
    for c in range(num_classes):
        X_c = class_matrices[c]
        X_c_centered = X_c - class_means[c] 
        S_W += np.dot(X_c_centered, X_c_centered.T)

    # handling singularity problem by adding beta*I in S_W where beta is very small scalor value
    # Regularization to ensure S_W is positive definite
    beta=1e-3
    S_W += beta * np.eye(S_W.shape[0])

    # Solve generalized eigenvalue problem: S_B W = λ S_W W
    eigvals, eigvecs = eigh(S_B, S_W)

    sorted_indices = np.argsort(eigvals)[::-1]
    W = eigvecs[:, sorted_indices]  # FDA projection matrix
    W = W[:, :num_classes - 1]  # Keeping only first (C-1) eigenvectors

    return W

# lda implementation
def train_lda(X_train, y_train):
    classes = np.unique(y_train)
    d, N = X_train.shape  
    C = len(classes)
    
    means = {}
    for c in classes:
        X_c = X_train[:, y_train == c]  
        mean_c = np.sum(X_c, axis=1, keepdims=True) / np.sum(y_train == c)
        means[c] = np.real(mean_c)  

    cov_matrix = np.zeros((d, d), dtype=np.float64)  
    for c in classes:
        X_c = X_train[:, y_train == c]  
        mean_c = means[c]  
        diff = X_c - mean_c
        cov_matrix += np.real(diff @ diff.T)  
    cov_matrix /= (N - C)  

    cov_matrix += 1e-6 * np.eye(d)  

    cov_inv = np.linalg.inv(cov_matrix).real  

    priors = {c: np.real(np.mean(y_train == c)) for c in classes}  

    return means, cov_inv, priors, classes

# classify data values based on discriminant function (x^T cov^-1 mu_c) - (0.5 u_c^T cov^-1 mu_c) + ln(p(c)) 
# :: data gets classified to the class which gives max value for this discriminant 
def predict_lda(X, means, cov_inv, priors, classes):
    predictions = []
    
    for i in range(X.shape[1]):  
        x = X[:, i:i+1]  
        g_x = {}
        
        for c in classes:
            mean_c = means[c]
            prior_c = np.log(priors[c]) if priors[c] > 0 else 0  
            
            gc_x = np.real(x.T @ cov_inv @ mean_c - 0.5 * mean_c.T @ cov_inv @ mean_c + prior_c)
            g_x[c] = gc_x[0, 0]  

        predictions.append(max(g_x, key=g_x.get))  
    
    return np.array(predictions)

# QDA implementation
def train_qda(X_train, labels):
    
    classes = np.unique(labels)
    d, N = X_train.shape
    means = {}
    covariances = {}
    priors = {}

    for c in classes:
        X_c = X_train[:, labels == c]  
        N_c = X_c.shape[1]  
        means[c] = np.sum(X_c, axis=1, keepdims=True)/N_c  # (d, 1)
        
        
        X_centered = X_c - means[c]  
        covariances[c] = (X_centered @ X_centered.T) / (N_c - 1)  # (d, d)
        
        priors[c] = N_c / N  

    return means, covariances, priors, classes

# classify data values based on discriminant function -0.5 (x^t cov^-1 x) + ( u_c^t cov^-1 x )- 0.5 (u_c^t cov^-1 u_x ) - 0.5 ln(det(cov)) + ln(p(c))  
# :: data gets classified to the class which gives max value for this discriminant 
def predict_qda(X_test, means, covariances, priors, classes):
    d, N = X_test.shape  # d = features, N = samples
    predictions = np.zeros(N, dtype=int) 
    
    cov_invs = {}
    logdets = {}   

    for c in classes:
        cov_c = covariances[c] + 1e-5 * np.eye(d)  
        cov_invs[c] = np.linalg.inv(cov_c)  
        sign, logdet = np.linalg.slogdet(cov_c)
        logdets[c] = logdet if sign > 0 else -np.inf  

    for j in range(N):
        x = X_test[:, j:j+1]  
        g_x = {}

        for c in classes:
            mean_c = means[c]  # (d, 1)
            delta = x - mean_c  # (d, 1)

            g_x[c] = -0.5 * delta.T @ cov_invs[c] @ delta - 0.5 * logdets[c] + np.log(priors[c])

        predictions[j] = max(g_x, key=g_x.get)

    return predictions


# Compute accuracy
def accuracy(y_true, y_pred):
    correct_count = 0
    total_count = len(y_true)

    for i in range(total_count):
        if y_true[i] == y_pred[i]:
            correct_count += 1

    return correct_count / total_count if total_count > 0 else 0.0


# Evaluate and Compare Performance

# Train LDA
lda_means,lda_cov_inv,lda_priors,lda_classes=train_lda(Train_Set_Images,Train_Set_Labels)

# Predict on train and test sets
lda_y_pred_train = predict_lda(Train_Set_Images, lda_means, lda_cov_inv, lda_priors, lda_classes)
lda_y_pred_test = predict_lda(Test_Set_Images, lda_means, lda_cov_inv, lda_priors, lda_classes)

lda_train_acc = accuracy(Train_Set_Labels, lda_y_pred_train)
lda_test_acc = accuracy(Test_Set_Labels, lda_y_pred_test)

print(f"LDA Train Accuracy: {lda_train_acc:.3f}")
print(f"LDA Test Accuracy: {lda_test_acc:.3f}")

# Train qda
qda_means, qda_covariances, qda_priors, qda_classes = train_qda(Train_Set_Images, Train_Set_Labels)

# Predict on train and test sets
qda_y_pred_train = predict_qda(Train_Set_Images, qda_means, qda_covariances, qda_priors, qda_classes)
qda_y_pred_test = predict_qda(Test_Set_Images, qda_means, qda_covariances, qda_priors, qda_classes)

qda_train_acc = accuracy(Train_Set_Labels, qda_y_pred_train)
qda_test_acc = accuracy(Test_Set_Labels, qda_y_pred_test)

print(f"QDA Train Accuracy: {qda_train_acc:.3f}")
print(f"QDA Test Accuracy: {qda_test_acc:.3f}")
print('\n')


# Apply FDA
print("\n" + "="*20 + " FDA " + "="*20 + "\n") 

W_fda = compute_FDA(Train_Set_Images, train_set_class_matrices, class_means, train_set_global_mean)

Y_test_FDA = W_fda.T @ Test_Set_Images  #(2x300) 
Y_train_FDA=W_fda.T @ Train_Set_Images #(2x300)

fda_train_plot=Y_train_FDA
fda_test_plot=Y_test_FDA

lda_means_fda, lda_cov_inv_fda, lda_priors_fda, lda_classes_fda = train_lda(Y_train_FDA, Train_Set_Labels)

lda_y_pred_train_fda = predict_lda(Y_train_FDA, lda_means_fda, lda_cov_inv_fda, lda_priors_fda, lda_classes_fda)
lda_y_pred_test_fda = predict_lda(Y_test_FDA, lda_means_fda, lda_cov_inv_fda, lda_priors_fda, lda_classes_fda)

lda_train_acc_fda = accuracy(Train_Set_Labels, lda_y_pred_train_fda)
lda_test_acc_fda = accuracy(Test_Set_Labels, lda_y_pred_test_fda)

print(f"LDA Train Accuracy after FDA: {lda_train_acc_fda:.3f}")
print(f"LDA Test Accuracy after FDA: {lda_test_acc_fda:.3f}")

qda_means_fda, qda_cov_inv_fda, qda_priors_fda, qda_classes_fda = train_qda(Y_train_FDA, Train_Set_Labels)

qda_y_pred_train_fda = predict_qda(Y_train_FDA, qda_means_fda, qda_cov_inv_fda, qda_priors_fda, qda_classes_fda)
qda_y_pred_test_fda = predict_qda(Y_test_FDA, qda_means_fda, qda_cov_inv_fda, qda_priors_fda, qda_classes_fda)

qda_Y_train_acc_fda = accuracy(Train_Set_Labels,qda_y_pred_train_fda)
qda_Y_test_acc_fda = accuracy(Test_Set_Labels,qda_y_pred_test_fda)

print(f"QDA Train Accuracy after FDA: {qda_Y_train_acc_fda:.3f}")
print(f"QDA Test Accuracy after FDA: {qda_Y_test_acc_fda:.3f}")

print('\n')

# Apply PCA
print("\n" + "="*20 + " PCA " + "="*20 + "\n")
U_p,p = compute_PCA(Train_Set_Images)
print(f"PCA applied: 95% variance retained")
print(f"PCA applied: {p} components retained")

# centralize matrices
X_train_centered=Train_Set_Images-train_set_global_mean
X_test_centered=Test_Set_Images-train_set_global_mean

Y_train_pca = U_p.T @ X_train_centered  
Y_test_pca = U_p.T @ X_test_centered    

lda_means_pca, lda_cov_inv_pca, lda_priors_pca, lda_classes_pca = train_lda(Y_train_pca, Train_Set_Labels)

# Predict on train and test sets for lda
lda_y_pred_train_pca = predict_lda(Y_train_pca, lda_means_pca, lda_cov_inv_pca, lda_priors_pca, lda_classes_pca)
lda_y_pred_test_pca = predict_lda(Y_test_pca, lda_means_pca, lda_cov_inv_pca, lda_priors_pca, lda_classes_pca)

lda_train_acc_pca = accuracy(Train_Set_Labels, lda_y_pred_train_pca)
lda_test_acc_pca = accuracy(Test_Set_Labels, lda_y_pred_test_pca)

print(f"LDA Train Accuracy after PCA: {lda_train_acc_pca:.3f}")
print(f"LDA Test Accuracy after PCA: {lda_test_acc_pca:.3f}")

# Predict on train and test sets for qda
qda_means_pca, qda_cov_inv_pca, qda_priors_pca, qda_classes_pca = train_qda(Y_train_pca, Train_Set_Labels)

qda_y_pred_train_pca = predict_qda(Y_train_pca, qda_means_pca, qda_cov_inv_pca, qda_priors_pca, qda_classes_pca)
qda_y_pred_test_pca = predict_qda(Y_test_pca, qda_means_pca, qda_cov_inv_pca, qda_priors_pca, qda_classes_pca)

qda_Y_train_acc_fda = accuracy(Train_Set_Labels,qda_y_pred_train_pca)
qda_Y_test_acc_fda = accuracy(Test_Set_Labels,qda_y_pred_test_pca)

# print(f"QDA Train Accuracy after PCA: {qda_Y_train_acc_fda:.3f}")
# print(f"QDA Test Accuracy after PCA: {qda_Y_test_acc_fda:.3f}")

print('\n')

print("\n" + "="*20 + " PCA " + "="*20 + "\n")

U_p,p = compute_PCA(Train_Set_Images,0.90)
print(f"PCA applied: 90% variance retained")
print(f"PCA applied: {p} components retained")

# centralize matrices
X_train_centered=Train_Set_Images-train_set_global_mean
X_test_centered=Test_Set_Images-train_set_global_mean

Y_train_pca = U_p.T @ X_train_centered  
Y_test_pca = U_p.T @ X_test_centered    

lda_means_pca, lda_cov_inv_pca, lda_priors_pca, lda_classes_pca = train_lda(Y_train_pca, Train_Set_Labels)

# Predict on train and test sets for lda
lda_y_pred_train_pca = predict_lda(Y_train_pca, lda_means_pca, lda_cov_inv_pca, lda_priors_pca, lda_classes_pca)
lda_y_pred_test_pca = predict_lda(Y_test_pca, lda_means_pca, lda_cov_inv_pca, lda_priors_pca, lda_classes_pca)

lda_train_acc_pca = accuracy(Train_Set_Labels, lda_y_pred_train_pca)
lda_test_acc_pca = accuracy(Test_Set_Labels, lda_y_pred_test_pca)

print(f"LDA Train Accuracy after PCA: {lda_train_acc_pca:.3f}")
print(f"LDA Test Accuracy after PCA: {lda_test_acc_pca:.3f}")

# Predict on train and test sets for qda
qda_means_pca, qda_cov_inv_pca, qda_priors_pca, qda_classes_pca = train_qda(Y_train_pca, Train_Set_Labels)

qda_y_pred_train_pca = predict_qda(Y_train_pca, qda_means_pca, qda_cov_inv_pca, qda_priors_pca, qda_classes_pca)
qda_y_pred_test_pca = predict_qda(Y_test_pca, qda_means_pca, qda_cov_inv_pca, qda_priors_pca, qda_classes_pca)

qda_Y_train_acc_fda = accuracy(Train_Set_Labels,qda_y_pred_train_pca)
qda_Y_test_acc_fda = accuracy(Test_Set_Labels,qda_y_pred_test_pca)

# print(f"QDA Train Accuracy after PCA: {qda_Y_train_acc_fda:.3f}")
# print(f"QDA Test Accuracy after PCA: {qda_Y_test_acc_fda:.3f}")

print('\n')
print("\n" + "="*20 + " PCA " + "="*20 + "\n")

U_p,p = compute_PCA(Train_Set_Images,0.95,2)
print(f"PCA applied: {p} components retained")

# centralize matrices
X_train_centered=Train_Set_Images-train_set_global_mean
X_test_centered=Test_Set_Images-train_set_global_mean

Y_train_pca = U_p.T @ X_train_centered  
Y_test_pca = U_p.T @ X_test_centered    

pca_train_plot=Y_train_pca
pca_test_plot=Y_test_pca

lda_means_pca, lda_cov_inv_pca, lda_priors_pca, lda_classes_pca = train_lda(Y_train_pca, Train_Set_Labels)

# Predict on train and test sets for lda
lda_y_pred_train_pca = predict_lda(Y_train_pca, lda_means_pca, lda_cov_inv_pca, lda_priors_pca, lda_classes_pca)
lda_y_pred_test_pca = predict_lda(Y_test_pca, lda_means_pca, lda_cov_inv_pca, lda_priors_pca, lda_classes_pca)

lda_train_acc_pca = accuracy(Train_Set_Labels, lda_y_pred_train_pca)
lda_test_acc_pca = accuracy(Test_Set_Labels, lda_y_pred_test_pca)

print(f"LDA Train Accuracy after PCA: {lda_train_acc_pca:.3f}")
print(f"LDA Test Accuracy after PCA: {lda_test_acc_pca:.3f}")

# Predict on train and test sets for qda
qda_means_pca, qda_cov_inv_pca, qda_priors_pca, qda_classes_pca = train_qda(Y_train_pca, Train_Set_Labels)

qda_y_pred_train_pca = predict_qda(Y_train_pca, qda_means_pca, qda_cov_inv_pca, qda_priors_pca, qda_classes_pca)
qda_y_pred_test_pca = predict_qda(Y_test_pca, qda_means_pca, qda_cov_inv_pca, qda_priors_pca, qda_classes_pca)

qda_Y_train_acc_fda = accuracy(Train_Set_Labels,qda_y_pred_train_pca)
qda_Y_test_acc_fda = accuracy(Test_Set_Labels,qda_y_pred_test_pca)

# print(f"QDA Train Accuracy after PCA: {qda_Y_train_acc_fda:.3f}")
# print(f"QDA Test Accuracy after PCA: {qda_Y_test_acc_fda:.3f}")

# Visualisation
fig,axs=plt.subplots(2,2)

# FDA Projection of Train Data
axs[0,0].scatter(fda_train_plot[0, Train_Set_Labels == 0], fda_train_plot[1, Train_Set_Labels == 0], label=f'Class {0}', c='red', marker='*')
axs[0,0].scatter(fda_train_plot[0, Train_Set_Labels == 1], fda_train_plot[1, Train_Set_Labels == 1], label=f'Class {1}', c='blue',marker='s')
axs[0,0].scatter(fda_train_plot[0, Train_Set_Labels == 2], fda_train_plot[1, Train_Set_Labels == 2], label=f'Class {2}', c='green',marker='o')
axs[0,0].set_title("FDA Projection of Train Data")
axs[0,0].legend()

# PCA Projection of Test Data
axs[0,1].scatter(fda_test_plot[0, Test_Set_Labels == 0], fda_test_plot[1, Test_Set_Labels == 0], label=f'Class {0}', c='red', marker='*')
axs[0,1].scatter(fda_test_plot[0, Test_Set_Labels == 1], fda_test_plot[1, Test_Set_Labels == 1], label=f'Class {1}', c='blue',marker='s')
axs[0,1].scatter(fda_test_plot[0, Test_Set_Labels == 2], fda_test_plot[1, Test_Set_Labels == 2], label=f'Class {2}', c='green',marker='o')
axs[0,1].set_title("FDA Projection of Test Data")
axs[0,1].legend()

# PCA Projection of Train Data
axs[1,0].scatter(pca_train_plot[0, Train_Set_Labels == 0], pca_train_plot[1, Train_Set_Labels == 0], label=f'Class {0}', c='red', marker='*')
axs[1,0].scatter(pca_train_plot[0, Train_Set_Labels == 1], pca_train_plot[1, Train_Set_Labels == 1], label=f'Class {1}', c='blue',marker='s')
axs[1,0].scatter(pca_train_plot[0, Train_Set_Labels == 2], pca_train_plot[1, Train_Set_Labels == 2], label=f'Class {2}', c='green',marker='o')
axs[1,0].set_title("PCA Projection of Train Data")
axs[1,0].legend()

# PCA Projection of Test Data
axs[1,1].scatter(pca_test_plot[0, Test_Set_Labels == 0], pca_test_plot[1, Test_Set_Labels == 0], label=f'Class {0}', c='red', marker='*')
axs[1,1].scatter(pca_test_plot[0, Test_Set_Labels == 1], pca_test_plot[1, Test_Set_Labels == 1], label=f'Class {1}', c='blue',marker='s')
axs[1,1].scatter(pca_test_plot[0, Test_Set_Labels == 2], pca_test_plot[1, Test_Set_Labels == 2], label=f'Class {2}', c='green',marker='o')
axs[1,1].set_title("PCA Projection of Test Data")
axs[1,1].legend()

fig.suptitle("PCA Projection of MNIST Digits (0,1,2)") 
fig.tight_layout()  
plt.get_current_fig_manager().full_screen_toggle()
plt.show()
