
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from kymatio.torch import Scattering2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math

'''matrix = np.random.rand(1000,1000)
shape = matrix.shape
gaussian_matrix = np.random.normal(loc=0.0, scale=2.0, size=shape)
print(gaussian_matrix.sum())
print(np.var(gaussian_matrix))'''

#%% 
###############################################
################ CONFIGURATION ################
###############################################

dataset_portions = [0.001, 0.002, 0.003]  # Portions of complete dataset for the accuracy vs dataset size
J, L, order = 4, 8, 2                          # Optimal values = log2(img_shape[1])-3,6-8; mopt=2
num_epochs_cuda, num_epochs_no_cuda = 200, 50  # How many epochs to run
batch_size = 32                                # For reading in data
model = ["scatterCNN", "normalCNN"][1]         # Artificial neural network
learning_rate = 1e-2          
regu = 1e-5       
noise_aver = 0
noise_std = 1         


###############################################
################# LOAD DATA ###################
###############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if torch.cuda.is_available(): device = "cuda"; num_epochs = 200; print(f"CUDA is available. Setting epochs to {num_epochs}.")
#else: device = "cpu"; num_epochs = 50; print(f"CUDA is not available. Setting epochs to {num_epochs}.")

print("Loading MNIST data...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Collect test images and labels for evaluation
test_images = []; test_labels = []
for images, labels in test_loader:
    for i in range(len(images)):
        test_images.append(images[i])
        test_labels.append(labels[i])

train_images = []; train_labels = []
for images, labels in train_loader:
    for i in range(len(images)): 
        train_images.append(images[i])
        train_labels.append(labels[i])


###############################################
############# DEFINE MODEL ####################
###############################################

class ScatteringModel(nn.Module):
    def __init__(self, J, L, order, input_shape, device):
        super(ScatteringModel, self).__init__()
        self.scattering = Scattering2D(J=J, L=L, max_order=order, shape=(28, 28)).to(device)

        # Use a dummy input to figure out the output shape of the scattering
        dummy_input = torch.zeros(1, *input_shape).to(device)
        dummy_scattering_output = self.scattering(dummy_input)

        # Flatten the scattering output for compatibility with the convolutional layers
        num_features = dummy_scattering_output.numel() / dummy_scattering_output.shape[0]
        self.flattened_size = int(num_features)

        # Setup convolutional layers expecting the flattened scattering output reshaped to (batch_size, num_features, 1, 1)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=1),  # Use 1x1 convolution to handle high-dimensional data
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),  # Avoid reducing dimension too much
            nn.Conv2d(6, 16, kernel_size=1),
            nn.ReLU()
        )

        # Dummy feature output to dynamically determine the size for fully connected layers
        dummy_feature_output = self.feature_extractor(torch.zeros(1, 1, self.flattened_size, 1).to(device))
        output_size = dummy_feature_output.view(dummy_feature_output.size(0), -1).shape[1]

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, 120),
            nn.ReLU(),
            nn.Linear(120, 10)  # Output for 10 classes
        )

    def forward(self, x):
        x = self.scattering(x)
        x = x.view(x.size(0), 1, self.flattened_size, 1)  # Reshape to include a single channel
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    
# Convolutional neural network without the scattering transform
class NormalModel(nn.Module):
    def __init__(self, input_shape):
        super(NormalModel, self).__init__()
        # Define basic convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # Assuming grayscale images as input
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Dummy input to calculate the output size after feature extraction
        dummy_input = torch.zeros(1, *input_shape)
        dummy_feature_output = self.feature_extractor(dummy_input)
        output_size = dummy_feature_output.view(dummy_feature_output.size(0), -1).shape[1]
        
        # Fully connected layers for multi-class classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Ten outputs for ten classes
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

 
if model == "scatterCNN": model = ScatteringModel(J, L, order, input_shape=(1, 28, 28), device=device); print("Scattering model selected")
elif model == "normalCNN": model = NormalModel(input_shape=(1, 28, 28)); print("Normal model selected")
else: ("Model not found error. Please selected one of 'scatterCNN' and 'normalCNN'")
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


###############################################
################ FIT MODEL ###################
###############################################

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

dataset_sizes = [int(len(train_images) * perc) for perc in dataset_portions]

accuracies = []
history = {'loss': [], 'val_loss': []}

for subset_size in dataset_sizes:
    subset_indices = np.random.choice(len(train_data), subset_size, replace=False)
    subset_train_data = Subset(train_data, subset_indices)
    
    train_indices, val_indices = train_test_split(np.arange(len(subset_train_data)), test_size=0.1, random_state=42)
    train_subset = Subset(subset_train_data, train_indices)
    val_subset = Subset(subset_train_data, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    num_epochs = 50 if subset_size < len(train_data) else 100
    
    for epoch in tqdm(range(num_epochs), desc=f'Training with dataset size {subset_size}'):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).to(device)
            #print(model.classifier.parameters())
            abs_weight_matrix = torch.abs(list(model.classifier.parameters())[2].data.detach()).to(device)
            l1_loss = regu * torch.sum(abs_weight_matrix).to(device)
            #out = list(model.classifier.parameters())[3].data
            #print(l1_loss)
            #print(out)
            loss = criterion(outputs, labels) + l1_loss
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        
        history['loss'].append(total_loss / len(train_subset))
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                #noise = torch.randn_like(images) * math.sqrt(1 / 10)
                shape = images.shape
                noise = np.random.normal(loc = noise_aver, scale = noise_std, size=shape)
                noisy_images = images + noise
                images = images.to(device)
                labels = labels.to(device)
                noisy_images = noisy_images.to(device).float()
                #outputs = model(images).to(device)
                outputs = model(noisy_images).to(device).float()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        history['val_loss'].append(val_loss / len(val_subset))
    
    # Evaluate and store the accuracy for the current subset after the training and validation loops
    accuracy = correct / total
    accuracies.append(accuracy)

    
#%%
###############################################
############ PRINT EVALUATIONS ################
###############################################

# Ensure test_images is a tensor and on the correct device
test_images_tensor = torch.stack(test_images).to(device)
outputs = model(test_images_tensor)
probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()  # Detach and then convert softmax probabilities to CPU and numpy array
pred_labels = probabilities.argmax(axis=1)  # Convert predictions to labels

# Continue with existing evaluation code
#print("trainlength = ", len(train_images))
#print("testlength = ", len(test_images))

# Calculate metric values
accuracy = accuracy_score(test_labels, pred_labels)
precision = precision_score(test_labels, pred_labels, average='macro')
recall = recall_score(test_labels, pred_labels, average='macro')
f1 = f1_score(test_labels, pred_labels, average='macro')

#auc = roc_auc_score(test_labels, probabilities, multi_class="ovr")
print(f"Test Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC: {auc}")

# Generate and plot the normalized confusion matrix
cm = confusion_matrix(test_labels, pred_labels, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2%", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Normalized Confusion Matrix')
plt.show()

def display_examples(X, y_true, y_pred, indices, title):
    plt.figure(figsize=(10, 3))
    for i, idx in enumerate(indices[:3]):  # Display first three examples
        plt.subplot(1, 3, i + 1)
        plt.imshow(X[idx].squeeze(), cmap='gray', interpolation='none')  # Make sure to squeeze in case there's an extra singleton dimension
        plt.title(f"{title}\nTrue: {y_true[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Calculate indices for true positives, false positives, true negatives, and false negatives
successful_indices = [i for i, (true, pred) in enumerate(zip(test_labels, pred_labels)) if true == pred]
unsuccessful_indices = [i for i, (true, pred) in enumerate(zip(test_labels, pred_labels)) if true != pred]

# Display examples of True Positives, False Positives, True Negatives, and False Negatives
display_examples(test_images, test_labels, pred_labels, successful_indices, title="Successful prediction")
display_examples(test_images, test_labels, pred_labels, unsuccessful_indices, title="Unsuccessful predicion")

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
