import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path
import pickle
import yaml

# Define the CNN model
class TransitionModel(nn.Module):
    def __init__(self, num_classes=10, param_dim=5):
        super(TransitionModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # Assuming input size 224x224
        self.param_fc = nn.Linear(param_dim, 128)  # FC layer for parameters
        self.fc2 = nn.Linear(512 + 128, num_classes)  # Concatenated features
        self.relu = nn.ReLU()

    def forward(self, x, params):

        # CNN for image
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = self.relu(self.fc1(x))
        
        # FC for params
        p = self.relu(self.param_fc(params))
        
        # Concatenate
        combined = torch.cat((x, p), dim=1)
        out = self.fc2(combined)
        return out

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

path = Path(__file__).parent
data_path = path / 'data/cubes'  # Adjust path as needed
subdirs = [d for d in data_path.iterdir() if d.is_dir()]
data_paths = [f for d in subdirs for f in d.glob('*.pkl')]
config_paths = [f for d in subdirs for f in d.glob('*.yaml')]

for c_path, s_path in zip(config_paths, data_paths):
    with open(c_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    params = [
        config_data['sandbox']['box']['properties']['friction'],
        config_data['sandbox']['material']['properties']['density'],
        config_data['sandbox']['material']['properties']['friction'],
    ]

    with open(s_path, 'rb') as f:
        samples = pickle.load(f)
    

    # Process config_data and sample_data as needed for training
    # For example, you might want to extract parameters and images from sample_data
    # and save them in a format suitable for the DataLoader

# Load data (assuming data is in genesis/data/ with subfolders for classes)
data_dir = '../data'  # Adjust path as needed
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer
model = TransitionModel(num_classes=len(train_dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train_model(model, train_loader, val_loader, criterion, optimizer)