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
import matplotlib.pyplot as plt



# Define the transformation
testtransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the dataset
test_root_folder = "/home/uday/anurag/data/test/"
testdataset = ImageFolder(root=test_root_folder, transform=testtransform)

# Create DataLoader
batch_size = 16
test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load the pre-trained VGG-16 model
vgg16 = models.vgg16(pretrained=False)  # Set pretrained to False since you are loading your own model
num_classes = 6
vgg16.classifier[6] = nn.Sequential(
    nn.Linear(vgg16.classifier[6].in_features, 4096),
    nn.ReLU(True),
    nn.Dropout(p=0.0),
    nn.Linear(4096, num_classes)
)

# Load the trained model weights
vgg16.load_state_dict(torch.load("/home/uday/anurag/98_87.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)


# Initialize dictionaries to track correct predictions and total instances for each class during training
correct_test_per_class = {class_idx: 0 for class_idx in range(num_classes)}
total_test_per_class = {class_idx: 0 for class_idx in range(num_classes)}


# Initialize variables to track correct predictions and total instances
correct_test = 0
total_test = 0

# Testing
vgg16.eval()

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)

        # Track training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        # Track training accuracy per class
        for class_idx in range(num_classes):
            class_mask = (labels == class_idx)
            correct_test_per_class[class_idx] += (predicted[class_mask] == labels[class_mask]).sum().item()
            total_test_per_class[class_idx] += class_mask.sum().item()

# Print correct predictions count and accuracy for each class during training



print("Testing Accuracy per Class:")
for class_idx in range(num_classes):
    class_accuracy = correct_test_per_class[class_idx] / total_test_per_class[class_idx] if total_test_per_class[class_idx] > 0 else 0
    print(f'Class {class_idx}: Correct Predictions: {correct_test_per_class[class_idx]}, Total Instances: {total_test_per_class[class_idx]}, Accuracy: {class_accuracy:.4f}')

accuracy_test = correct_test / total_test    

# Print overall correct predictions
print(f'Overall Correct Predictions: {correct_test}, Total Instances: {total_test}')
print("final test accuracy is :",accuracy_test)