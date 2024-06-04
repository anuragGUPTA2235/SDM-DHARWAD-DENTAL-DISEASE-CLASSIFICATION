# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from tqdm import tqdm

# # Define the transformation
# testtransform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Define the dataset
# test_root_folder = "/home/ml3gpu/resnet/old_dataset/test1"
# testdataset = ImageFolder(root=test_root_folder, transform=testtransform)

# # Create DataLoader
# batch_size = 6
# test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=4)

# # Load pre-trained VGG-16 model
# vgg16 = models.vgg16(pretrained=True)
# num_classes = 4
# vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, num_classes)

# # Load the trained model
# vgg16.load_state_dict(torch.load("/home/ml3gpu/resnet/vgg16(6class)data1.pth"))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vgg16.to(device)

# # Initialize dictionaries to track correct predictions and total instances for each class during testing
# correct_test_per_class = {class_idx: 0 for class_idx in range(num_classes)}
# total_test_per_class = {class_idx: 0 for class_idx in range(num_classes)}

# # Testing
# vgg16.eval()
# correct_test = 0
# total_test = 0

# with torch.no_grad():
#     for images, labels in tqdm(test_loader):
#         images, labels = images.to(device), labels.to(device)
#         outputs = vgg16(images)

#         # Track testing accuracy
#         _, predicted = torch.max(outputs.data, 1)
#         total_test += labels.size(0)
#         correct_test += (predicted == labels).sum().item()

#         # Track testing accuracy per class
#         for class_idx in range(num_classes):
#             class_mask = (labels == class_idx)
#             correct_test_per_class[class_idx] += (predicted[class_mask] == labels[class_mask]).sum().item()
#             total_test_per_class[class_idx] += class_mask.sum().item()

# # Print correct predictions count and accuracy for each class
# for class_idx in range(num_classes):
#     class_accuracy = correct_test_per_class[class_idx] / total_test_per_class[class_idx] if total_test_per_class[class_idx] > 0 else 0
#     print(f'Class {class_idx}: Correct Predictions: {correct_test_per_class[class_idx]}, Total Instances: {total_test_per_class[class_idx]}, Accuracy: {class_accuracy:.4f}')

# accuracy_test = correct_test / total_test

# # Print final test accuracy
# print(f'Final Testing Accuracy: {accuracy_test:.4f}')




import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Define the transformation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the dataset
train_root_folder = "/home/uday/anurag/data/train/"
train_dataset = ImageFolder(root=train_root_folder, transform=train_transform)

# Create DataLoader with error handling
batch_size = 6
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Load pre-trained VGG-16 model
vgg16 = models.vgg16(pretrained=True)
num_classes = len(train_dataset.classes)
vgg16.classifier[6] = nn.Sequential(
    nn.Linear(vgg16.classifier[6].in_features, 4096),
    nn.ReLU(True),
    nn.Dropout(p=0.0),
    nn.Linear(4096, num_classes)
)


# Load the trained model
vgg16.load_state_dict(torch.load("/home/uday/anurag/98_87.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)

# Initialize dictionaries to track correct predictions and total instances for each class during training
correct_train_per_class = {class_idx: 0 for class_idx in range(num_classes)}
total_train_per_class = {class_idx: 0 for class_idx in range(num_classes)}

# Training
vgg16.eval()  # Ensure the model is in evaluation mode for the training loop
correct_train = 0
total_train = 0

with torch.no_grad():
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)

        # Track training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Track training accuracy per class
        for class_idx in range(num_classes):
            class_mask = (labels == class_idx)
            correct_train_per_class[class_idx] += (predicted[class_mask] == labels[class_mask]).sum().item()
            total_train_per_class[class_idx] += class_mask.sum().item()

# Print correct predictions count and accuracy for each class during training
print("Training Accuracy per Class:")
for class_idx in range(num_classes):
    class_accuracy = correct_train_per_class[class_idx] / total_train_per_class[class_idx] if total_train_per_class[class_idx] > 0 else 0
    print(f'Class {class_idx}: Correct Predictions: {correct_train_per_class[class_idx]}, Total Instances: {total_train_per_class[class_idx]}, Accuracy: {class_accuracy:.4f}')

accuracy_train = correct_train / total_train

# Print final training accuracy
print(f'Final Training Accuracy: {accuracy_train:.4f}')



class_accuracies = [(correct_train_per_class[class_idx] / total_train_per_class[class_idx])*100 if total_train_per_class[class_idx] > 0 else 0 for class_idx in range(num_classes)]
ylabels = ['3R','3L','2R','2L','1R','1L']
xlabels = class_accuracies
print(xlabels)
print(ylabels)
plt.figure(figsize=(10, 5))
bars = plt.bar(ylabels, class_accuracies, color='pink')
plt.xlabel('Dataset Classes')
plt.ylabel('Train Accuracy')
plt.title('Class-wise Accuracy')
plt.xticks(range(num_classes))
plt.ylim(0, 100)

for bar, accuracy in zip(bars, class_accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, 
             bar.get_height() / 2,  # Center text vertically at half the height of the bar
             f'{accuracy:.2f}%', 
             ha='center', va='center')  # Center text horizontally and vertically

# Save the plot
plt.savefig('class_wise_train.png')



from sklearn.metrics import confusion_matrix
import seaborn as sns

# Initialize the confusion matrix
conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)


# Calculate the confusion matrix
with torch.no_grad():
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs, 1)
        
        # Update the confusion matrix
        conf_matrix += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=range(num_classes))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Oranges", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Train Confusion Matrix')
plt.savefig('train_con_matrix.png')

