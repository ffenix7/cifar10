    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision 
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    # === MODEL ===
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)

            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)

            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)

            self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(512)

            self.pool = nn.MaxPool2d(2)

            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            x = self.gap(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    # === TRANSFORMACJE ===
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    # === DANE ===
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # === MODEL I OPT ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # === TRENING ===
    def train_model(model, trainloader, testloader, optimizer, scheduler, criterion, device, epochs):
        train_losses, test_accuracies = [], []
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(trainloader)
            train_losses.append(avg_loss)

            # Ewaluacja
            accuracy = test_model(model, testloader, device)
            test_accuracies.append(accuracy)
            scheduler.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return train_losses, test_accuracies

    # === TEST ===
    def test_model(model, testloader, device):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    # === WYKONANIE ===
    train_losses, test_accuracies = train_model(model, trainloader, testloader, optimizer, scheduler, criterion, device, epochs=30)

    # === WYKRESY ===
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Strata treningowa')
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='orange')
    plt.title('Dokładność na zbiorze testowym')
    plt.xlabel('Epoka')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
