import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold

os.chdir("C:/Users/송채영/Desktop/송채영/HAI/code/model")

model_config = np.array([[16, 32, 64],    # channel each convolution layer
                         [2, 2, 2],       # kernel size of convolution layer
                         [1, 1, 1],       # stride size of convolution layer
                         [2, 2, 2],       # kernel size of pooling layer
                         [1, 1, 1],       # stride size of pooling layer
                         [64, 32, 16],    # channel of dense block layer
                         [4, 4, 4]        # classes
                         ])

class Conv2DModel(nn.Module):
    def __init__(self, channel_num, paramArr):
        super(Conv2DModel, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num, out_channels=paramArr[0][0], kernel_size=(2, paramArr[1][0]), stride=(1, paramArr[2][0]), padding=2, bias=False),
            nn.BatchNorm2d(paramArr[0][0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, paramArr[3][0]), stride=(1, paramArr[4][0]))
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=paramArr[0][0], out_channels=paramArr[0][1], kernel_size=(2, paramArr[1][1]), stride=(1, paramArr[2][1]), bias=False),
            nn.BatchNorm2d(paramArr[0][1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, paramArr[3][1]), stride=(1, paramArr[4][1]))
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=paramArr[0][1], out_channels=paramArr[0][2], kernel_size=(2, paramArr[1][2]), stride=(1, paramArr[2][2]), bias=False),
            nn.BatchNorm2d(paramArr[0][2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, paramArr[3][2]), stride=(1, paramArr[4][2]))
        )

        self.flatten = nn.Flatten()
        self.denseblock = nn.Sequential(
            nn.LazyLinear(paramArr[5][0]),
            nn.Linear(paramArr[5][0], paramArr[5][1]),
            nn.Linear(paramArr[5][1], paramArr[5][2]),
            nn.ReLU()
        )
        #self.out = nn.Linear(1728, paramArr[6][0])
        self.out = nn.Linear(paramArr[5][2], paramArr[6][0])
        #self.out = nn.Linear(864, paramArr[6][0])

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.flatten(x)
        x = self.denseblock(x)
        x = self.out(x)
        return x

def sample_data_from_segments(data, labels, num_samples_per_segment=5):
    sampled_data = []
    sampled_labels = []
    segment_indices = [
        # range(0, 800),       # Segment 1: indices 1 to 800 (0-indexed)
        # range(800, 1600),    # Segment 2: indices 801 to 1600
        # range(1600, 2400),   # Segment 3: indices 1601 to 2400
        # range(2400, 3200)    # Segment 4: indices 2401 to 3200

        #5sec
        # range(0,160),
        # range(160,320),
        # range(320,480),
        # range(480,640)

        #10sec
        range(0, 80),
        range(80, 160),
        range(160, 240),
        range(240, 320)
    ]
    for segment in segment_indices:
        selected_indices = np.random.choice(segment, num_samples_per_segment, replace=False)
        
        sampled_data.append(data[selected_indices])
        sampled_labels.append(labels[selected_indices])

    return np.concatenate(sampled_data), np.concatenate(sampled_labels)

# Load and preprocess data
def load_data(file_path, num_samples_per_segment=10):
    full_data = np.load(file_path)
    data = full_data[:, :-1] # Reshape
    labels = full_data[:, -1]
    labels[labels==3] = 0
    
    # Normalize data(z-score)
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1
    data = (data - mean) / std
    
    data = data.reshape(320, 1, 4, 7)

    return sample_data_from_segments(data, labels, num_samples_per_segment)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

all_data, all_labels = [], []

for i in range(1, 51):
    data, labels = load_data(f"{i}.npy", 80)
    all_data.append(data)
    all_labels.append(labels)

# Convert to tensors
all_data = torch.tensor(np.concatenate(all_data), dtype=torch.float32)
all_labels = torch.tensor(np.concatenate(all_labels), dtype=torch.int64)

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion):
    num_epochs = 30
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    all_targets = []
    all_predictions = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_targets.extend(targets.tolist())
                all_predictions.extend(predicted.tolist())

        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies, all_targets, all_predictions

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

group_labels = np.array([i for i in range(1, 51) for _ in range(320)])

kf = GroupKFold(n_splits=5)

# Initialize lists to hold average metrics for all folds
all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []
overall_confusion_matrix = np.zeros((3, 3))  # Initialize confusion matrix for 4 classes

# 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(all_data, all_labels, groups=group_labels)):
    print(f"Fold {fold+1}")

    train_data = all_data[train_idx]
    train_labels = all_labels[train_idx]
    test_data = all_data[test_idx]
    test_labels = all_labels[test_idx]

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Conv2DModel(1, model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train and evaluate model for the current fold
    train_losses, val_losses, train_accuracies, val_accuracies, fold_targets, fold_predictions = train_and_evaluate(model, train_loader, test_loader, optimizer, criterion)

    # Append the metrics for this fold
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_train_accuracies.append(train_accuracies)
    all_val_accuracies.append(val_accuracies)

    # Update overall confusion matrix
    fold_cm = confusion_matrix(fold_targets, fold_predictions, labels=[0, 1, 2])
    overall_confusion_matrix += fold_cm

# Calculate average metrics across all folds
average_train_losses = np.mean(all_train_losses, axis=0)
average_val_losses = np.mean(all_val_losses, axis=0)
average_train_accuracies = np.mean(all_train_accuracies, axis=0)
average_val_accuracies = np.mean(all_val_accuracies, axis=0)

# Normalize the confusion matrix
normalized_confusion_matrix = overall_confusion_matrix / overall_confusion_matrix.sum(axis=1, keepdims=True)
normalized_confusion_matrix = np.nan_to_num(normalized_confusion_matrix)  # Replace NaNs with 0

# Print overall test accuracy and loss across all folds
overall_test_accuracy = np.mean([acc[-1] for acc in all_val_accuracies])  # Test accuracy for the last epoch in each fold
overall_test_loss = np.mean([loss[-1] for loss in all_val_losses])  # Test loss for the last epoch in each fold
print(f"\nTest Accuracy : {overall_test_accuracy:.2f}%")
print(f"Test Loss : {overall_test_loss:.4f}\n")

# Plotting average train/val loss and accuracy after all folds
num_epochs = len(average_train_losses)
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(14, 5))

# Plot average loss
plt.subplot(1, 2, 1)
plt.plot(epochs, average_train_losses, label='Average Train Loss', marker='o')
plt.plot(epochs, average_val_losses, label='Average Validation Loss', marker='o')
plt.title('Average Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot average accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, average_train_accuracies, label='Average Train Accuracy', marker='o')
plt.plot(epochs, average_val_accuracies, label='Average Validation Accuracy', marker='o')
plt.title('Average Train and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the confusion matrix
plot_confusion_matrix(normalized_confusion_matrix, classes=['supine & prone', 'left', 'right'])
