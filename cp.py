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

"""
class_accuracies = [(correct_test_per_class[class_idx] / total_test_per_class[class_idx])*100 if total_test_per_class[class_idx] > 0 else 0 for class_idx in range(num_classes)]

ylabels = ['3R','3L','2R','2L','1R','1L']
xlabels = class_accuracies
print(xlabels)
print(ylabels)


plt.figure(figsize=(10, 5))
plt.bar(range(num_classes), class_accuracies, color='green')
plt.xlabel('Class Index')
plt.ylabel('test Accuracy')
plt.title('Class-wise Accuracy')
plt.xticks(range(num_classes))
plt.ylim(0, 1)
plt.grid(axis='y')

# Save the plot
plt.savefig('class_wise_accuracy_test.png')

"""
"""

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Initialize the confusion matrix
conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)


# Calculate the confusion matrix
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs, 1)
        
        # Update the confusion matrix
        conf_matrix += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=range(num_classes))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Test Confusion Matrix')
plt.savefig('test_con_matrix.png')
"""

"""
from sklearn.metrics import roc_auc_score,roc_curve
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you have defined your test_loader and vgg16 model

# Initialize lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)

        _, predicted = torch.max(outputs, 1)
        
        # Convert labels to one-hot encoding
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        
        # Calculate probability scores from the output logits
        softmax_scores = torch.nn.functional.softmax(outputs, dim=1)
        
        # Append true labels and predicted probabilities for ROC AUC calculation
        true_labels.extend(one_hot_labels.cpu().numpy())
    
        predicted_probs.extend(softmax_scores.cpu().numpy())
    

# Calculate ROC AUC score
"""
#roc_auc = roc_auc_score(true_labels, predicted_probs, multi_class='ovr')

"""

print("ROC AUC Score:", roc_auc)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import numpy as np



# Convert true labels to one-hot encoding
true_labels_one_hot = label_binarize(true_labels, classes=np.unique(true_labels))
print(true_labels_one_hot)

# Compute ROC curve and ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(true_labels_one_hot.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(true_labels_one_hot[:, i], predicted_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(true_labels_one_hot.shape[1]):
    plt.plot(fpr[i], tpr[i], label='ROC curve (class {}) (AUC = {:.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Plot diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multiclass Classification')
plt.legend(loc="lower right")
plt.savefig("roc_auc_curve.png")
"""



        









