import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler
from torchsummary import summary
from tqdm import tqdm
import numpy as np

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
testtransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Define the dataset
test_root_folder = "test1"
root_folder = 'train1'
dataset = ImageFolder(root=root_folder, transform=transform)
testdataset = ImageFolder(root=test_root_folder, transform=testtransform)

# Create DataLoader
batch_size = 6
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load pre-trained VGG-16 model
vgg16 = models.vgg16(pretrained=True)

# Modify the last fully connected layer for the new number of classes
num_classes = 4
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, num_classes)

# Move the model to the same device as the input data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)

# Define the loss function and optimizer for training
criterion = nn.CrossEntropyLoss()
weight_decay = 0.001
optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay)

# Training loop
num_epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)

epoch_list = []
train_accuracy_list = []
test_accuracy_list = []

max_accuracy = 0.0  # Variable to store the maximum testing accuracy

for epoch in range(num_epochs):
    # Training
    vgg16.train()
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = vgg16(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    accuracy_train = correct_train / total_train

    # Testing
    vgg16.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            loss = criterion(outputs, labels)

            # Track testing accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    accuracy_test = correct_test / total_test

    epoch_list.append(epoch)
    train_accuracy_list.append(accuracy_train)
    test_accuracy_list.append(accuracy_test)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {accuracy_train:.4f}, Testing Accuracy: {accuracy_test:.4f}')

    # Update max_accuracy if a new maximum is achieved
    if accuracy_test > max_accuracy:
        max_accuracy = accuracy_test
        # Save the model with maximum testing accuracy
        torch.save(vgg16.state_dict(), 'vgg16_max(4class)data.pth')

print(f'Maximum Testing Accuracy: {max_accuracy:.4f}')

# Print model summary
summary(vgg16, input_size=(3, 224, 224))

torch.save(vgg16.state_dict(), 'vgg16(4class)data.pth')

# Convert lists to numpy arrays
epoch_array = np.array(epoch_list)
train_accuracy_array = np.array(train_accuracy_list)
test_accuracy_array = np.array(test_accuracy_list)

# Print arrays of epochs, training accuracies, and testing accuracies
print("Epochs Array:")
print(epoch_array)
print("\nTraining Accuracies Array:")
print(train_accuracy_array)
print("\nTesting Accuracies Array:")
print(test_accuracy_array)
