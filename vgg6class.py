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
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt


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
test_root_folder = "/home/uday/anurag/data/test/"
root_folder = '/home/uday/anurag/data/train/'
dataset = ImageFolder(root=root_folder, transform=transform)
testdataset = ImageFolder(root=test_root_folder, transform=testtransform)

# Create DataLoader
batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load pre-trained VGG-16 model
vgg16 = models.vgg16(pretrained=True)

# Modify the last fully connected layer for the new number of classes
num_classes = 6
# Modify the last fully connected layer for the new number of classes and add dropout
vgg16.classifier[6] = nn.Sequential(
    nn.Linear(vgg16.classifier[6].in_features, 4096),
    nn.ReLU(True),
    nn.Dropout(p=0.8),
    nn.Linear(4096, num_classes)
)


# Move the model to the same device as the input data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)

# Define the loss function and optimizer for training
criterion = nn.CrossEntropyLoss()
weight_decay = 0.001
lr_rate = 0.000001
#optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay)

num_epochs = 100
optimizer = torch.optim.AdamW(vgg16.parameters(), lr=lr_rate, weight_decay=0.01)
lr_scheduler= CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Training loop

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
        lr_scheduler.step()

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
        torch.save(vgg16.state_dict(), 'dental_max.pth')

print(f'Maximum Testing Accuracy: {max_accuracy:.4f}')

# Print model summary
summary(vgg16, input_size=(3, 224, 224))

torch.save(vgg16.state_dict(), 'vgg16(6class)data.pth')

# Convert lists to numpy arrays
epoch_array = np.array(epoch_list)
train_accuracy_array = np.array(train_accuracy_list)
test_accuracy_array = np.array(test_accuracy_list)

train_accuracy_array *= 100
test_accuracy_array *= 100

# Print arrays of epochs, training accuracies, and testing accuracies
print("Epochs Array:")
print(epoch_array)
print("\nTraining Accuracies Array:")
print(train_accuracy_array)
print("\nTesting Accuracies Array:")
print(test_accuracy_array)


# Plotting
plt.figure(figsize=(10, 5))
plt.plot(epoch_array, train_accuracy_array, label='Training Accuracy')
plt.plot(epoch_array, test_accuracy_array, label='Testing Accuracy')
plt.title('Training and Testing Accuracies vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('dental.png')

# Display the plot
plt.show()





# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# import torchvision.transforms as transforms
# from tqdm import tqdm
# import numpy as np
# from torchsummary import summary

# # Define transform for resizing, color jitter, normalization
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 406], std=[0.229, 0.224, 0.225]),
# ])

# # Define dataset and data loaders
# test_root_folder = "test1"
# root_folder = 'train1'
# dataset = ImageFolder(root=root_folder, transform=transform)
# testdataset = ImageFolder(root=test_root_folder, transform=transform)

# # Create DataLoader
# batch_size = 6
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=4)

# # Define CustomVGG16 model (corrected convolutional layer)
# class CustomVGG16(nn.Module):
#     def __init__(self, num_classes=6):
#         super(CustomVGG16, self).__init__()

#         # Initial convolutional layers (same as original)
#         self.initial_conv_layers = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Input channels adjusted to 3
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         # Additional convolutional layers for downsampling
#         self.downsampling_layers = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         # Rest of the VGG-16 layers (same as original)
#         vgg16 = models.vgg16(pretrained=True)
#         self.features = nn.Sequential(*list(vgg16.features.children())[7:])

#         # **Crucial Correction:** Ensure compatibility between input features and linear layer
#         # (code remains the same as explained before)
#         num_features = self._get_num_features(self.features)
#         self.classifier = nn.Linear(num_features, num_classes)

#     def _get_num_features(self, feature_extractor):
#         # Function to calculate the number of output features of a sequential layer
#         # (can be adapted for more complex architectures)
#         x = torch.randn(1, 3, 512, 512)  # Create a dummy input tensor with 3 channels
#         output = feature_extractor(x)
#         return output.view(1, -1).shape[1]  # Get the number of features from the output shape

#     def forward(self, x):
#         x = self.initial_conv_layers(x)
#         x = self.downsampling_layers(x)
#         x = self.features(x)
#         x = x.view(x.size(0), -1)  # Flatten features before linear layer
#         x = self.classifier(x)
#         return x


# # Create an instance of the CustomVGG16 model
# num_classes = 6
# custom_vgg16 = CustomVGG16(num_classes=num_classes)

# # Move the model to the same device as the input data
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# custom_vgg16.to(device)

# # Define the loss function and optimizer for training
# criterion = nn.CrossEntropyLoss()
# weight_decay = 0.001
# optimizer = torch.optim.SGD(custom_vgg16.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay)

# # Training loop
# num_epochs = 150

# epoch_list = []
# train_accuracy_list = []
# test_accuracy_list = []

# max_accuracy = 0.0  # Variable to store the maximum testing accuracy

# for epoch in range(num_epochs):
#     # Training
#     custom_vgg16.train()
#     correct_train = 0
#     total_train = 0

#     for images, labels in tqdm(train_loader):
#         images, labels = images.to(device), labels.to(device)
        
#         # Forward pass
#         outputs = custom_vgg16(images)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Track training accuracy
#         _, predicted = torch.max(outputs.data, 1)
#         total_train += labels.size(0)
#         correct_train += (predicted == labels).sum().item()

#     accuracy_train = correct_train / total_train

#     # Testing
#     custom_vgg16.eval()
#     correct_test = 0
#     total_test = 0

#     with torch.no_grad():
#         for images, labels in tqdm(test_loader):
#             images, labels = images.to(device), labels.to(device)
#             outputs = custom_vgg16(images)
#             loss = criterion(outputs, labels)

#             # Track testing accuracy
#             _, predicted = torch.max(outputs.data, 1)
#             total_test += labels.size(0)
#             correct_test += (predicted == labels).sum().item()
    
#     accuracy_test = correct_test / total_test

#     epoch_list.append(epoch)
#     train_accuracy_list.append(accuracy_train)
#     test_accuracy_list.append(accuracy_test)

#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {accuracy_train:.4f}, Testing Accuracy: {accuracy_test:.4f}')

#     # Update max_accuracy if a new maximum is achieved
#     if accuracy_test > max_accuracy:
#         max_accuracy = accuracy_test
#         # Save the model with maximum testing accuracy
#         torch.save(custom_vgg16.state_dict(), 'custom_vgg16_max(6class)conv.pth')

# print(f'Maximum Testing Accuracy: {max_accuracy:.4f}')

# # Print model summary
# summary(custom_vgg16, input_size=(3, 512, 512))

# # Save the final trained model
# torch.save(custom_vgg16.state_dict(), 'custom_vgg16(6class)conv.pth')

# # Convert lists to numpy arrays
# epoch_array = np.array(epoch_list)
# train_accuracy_array = np.array(train_accuracy_list)
# test_accuracy_array = np.array(test_accuracy_list)

# # Print arrays of epochs, training accuracies, and testing accuracies
# print("Epochs Array:")
# print(epoch_array)
# print("\nTraining Accuracies Array:")
# print(train_accuracy_array)
# print("\nTesting Accuracies Array:")
# print(test_accuracy_array)

