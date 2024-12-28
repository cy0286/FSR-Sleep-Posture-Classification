import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import timm
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())

# os.chdir('/home/user3/다운로드') 
os.chdir("C:/Users/송채영/Desktop/송채영/HAI/code/")

# Log file path
# log_file_path = "/home/user3/다운로드/plots/training_log.txt"
log_file_path = "C:/Users/송채영/Desktop/송채영/HAI/code/model/training_log.txt"

# List of ViT models with 224 input size
vit_models = [
    'vit_tiny_patch16_224'
]

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = os.listdir(image_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder + '/', self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = int((int(self.images[idx].split('_')[2].split('.')[0])-1) // 80)
        # if label == 0 or label == 3:
        #     label = 0
        if self.transform:
            image = self.transform(image)

        return image, label

# Define image transformations
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Create datasets and dataloaders
train_dataset = ImageDataset('./plots/train', transform=transform)
valid_dataset = ImageDataset('./plots/valid', transform=transform)
test_dataset = ImageDataset('./plots/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop for each ViT model
with open(log_file_path, 'a') as log_file:
    for model_name in vit_models:
        print(f"Training model: {model_name}")
        log_file.write(f"Training model: {model_name}\n")

        # Initialize model
        model = timm.create_model(model_name, pretrained=True)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        num_epochs = 5

        # Variables to track the best accuracy and corresponding epoch
        best_val_acc = 0.0
        best_epoch = -1
        best_model_weights = None

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            scheduler.step()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total

            # Validation loop
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in tqdm(valid_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(valid_loader.dataset)
            val_acc = val_correct / val_total

            # Log the results
            log_file.write(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}\n")
            log_file.write(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

            # Check if this is the best epoch so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_weights = model.state_dict()  # Save the best model weights

                # Save the best model weights with torch.save()
                save_path = f'best_model_{model_name}_epoch{best_epoch}.pth'
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc
                }, save_path)
            
                print(f"Best model saved at epoch {best_epoch} with validation accuracy {best_val_acc:.4f}")
                log_file.write(f"Best model saved at epoch {best_epoch} with validation accuracy {best_val_acc:.4f}\n")

        # Test the model using the best weights
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)  # Load the best model weights for testing
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in tqdm(test_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            test_loss /= len(test_loader.dataset)
            test_acc = test_correct / test_total

            log_file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n\n")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")