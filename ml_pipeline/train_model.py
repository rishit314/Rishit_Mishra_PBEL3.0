import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configuration
data_dir = './ml_pipeline/dataset' # Point this to your folder containing 'train' and 'val'
batch_size = 32
num_classes = 38 # Ensure this matches your directory structure

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(f'{data_dir}/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Quick training loop (Just 1-2 epochs for testing)
model.train()
for epoch in range(3):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3", leave=True)
    
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    print(f"Epoch {epoch+1} Complete. Average Loss: {running_loss / len(train_loader):.4f}")

# Save the weights
torch.save(model.state_dict(), "best_model.pth")
print("Training complete. 'best_model.pth' saved.")