import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Config
data_dir = "training_data"  # Ruta al dataset
batch_size = 8
num_epochs = 5
model_path = "logo_classifier.pth"

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# División en train/val
train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

# Modelo
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 clases: logo / no_logo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("🚀 Entrenando modelo...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Época {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

# Validación básica
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"✅ Precisión en validación: {100 * correct / total:.2f}%")

# Guardar modelo
torch.save(model.state_dict(), model_path)
print(f"📦 Modelo guardado en: {model_path}")
