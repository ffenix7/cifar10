import torch
import torchvision
import torchvision.transforms as transforms
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# === TRANSFORMATION ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# === LOAD DATASET (train on a sample) ===
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Sample a small subset
def get_data(dataset, n_samples):
    loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=True)
    images, labels = next(iter(loader))
    images = images.view(images.size(0), -1)  # flatten
    return images.numpy(), labels.numpy()

X_train, y_train = get_data(trainset, 100000)
X_test, y_test = get_data(testset, 1000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === PCA ===
pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# === TRAINING ===
clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
print("Training...")
clf.fit(X_train_pca, y_train)

# === TESTING ===
y_pred = clf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# === PLOT: explained variance by PCA ===
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Cumulative Explained Variance by PCA")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

# === PCA to 2D for visualization ===
pca_2d = PCA(n_components=2)
X_vis = pca_2d.fit_transform(X_train)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, cmap='tab10', alpha=0.5, s=10)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("Training Data after PCA (2 Components)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.show()

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=trainset.classes, yticklabels=trainset.classes)
plt.xlabel("Prediction")
plt.ylabel("True Label")
plt.title("Confusion Matrix - SVM on CIFAR-10")
plt.tight_layout()
plt.show()
