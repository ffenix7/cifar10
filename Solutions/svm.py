import torch
import torchvision
import torchvision.transforms as transforms
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# === TRANSFORMACJA (brak normalizacji, tylko tensor i flatten później) ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

# === ZAŁADUJ ZBIÓR (trenuj na próbce) ===
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Próbkuj mały podzbiór (np. 5000 próbek) — pełen CIFAR-10 zajmuje za dużo czasu
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
# Zmniejszamy wymiarowość do 300, aby przyspieszyć SVM
pca = PCA(n_components=300)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# === SVM ===
clf = svm.SVC(kernel='rbf', C=10, gamma='scale')  # możesz pobawić się hyperparametrami
print("Trenowanie SVM...")
clf.fit(X_train, y_train)

# === TEST ===
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność na zbiorze testowym: {accuracy * 100:.2f}%")
