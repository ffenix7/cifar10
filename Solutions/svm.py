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

# === TRANSFORMACJA ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# === ZAŁADUJ ZBIÓR (trenuj na próbce) ===
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Próbkuj mały podzbiór
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

# === WYKRES: wyjaśniona wariancja PCA ===
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Skumulowana wyjaśniona wariancja przez PCA")
plt.xlabel("Liczba komponentów")
plt.ylabel("Skumulowana wariancja")
plt.grid(True)
plt.tight_layout()
plt.show()

# === PCA do 2D dla wizualizacji danych ===
pca_2d = PCA(n_components=2)
X_vis = pca_2d.fit_transform(X_train)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, cmap='tab10', alpha=0.5, s=10)
plt.legend(*scatter.legend_elements(), title="Klasy")
plt.title("Dane treningowe po PCA (2 komponenty)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.show()

# === TRENING ===
clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
print("Trenowanie")
clf.fit(X_train_pca, y_train)

# === TEST ===
y_pred = clf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność na zbiorze testowym: {accuracy * 100:.2f}%")

# === MACIERZ POMYŁEK ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=trainset.classes, yticklabels=trainset.classes)
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa etykieta")
plt.title("Macierz pomyłek - SVM na CIFAR-10")
plt.tight_layout()
plt.show()