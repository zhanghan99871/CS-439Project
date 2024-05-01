# %%
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
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm

'''matrix = np.random.rand(1000,1000)
shape = matrix.shape
gaussian_matrix = np.random.normal(loc=0.0, scale=2.0, size=shape)
print(gaussian_matrix.sum())
print(np.var(gaussian_matrix))'''

# %%
###############################################
################ CONFIGURATION ################
###############################################

dataset_portions = [0.0001, 0.0002, 0.0003]  # Portions of complete dataset for the accuracy vs dataset size
#dataset_portions = [1e-4, 1e-3, 1e-2, 1e-1, 1]  # Portions of complete dataset for the accuracy vs dataset size
num_experiments = 3
num_epochs_cuda, num_epochs_no_cuda = 200, 5  # How many epochs to run
batch_size = 32  # For reading in data

J, L, order = 4, 8, 2  # Optimal values = log2(img_shape[1])-3,6-8; mopt=2
CNNmodel = ["scatterCNN", "normalCNN"][1]  #Pick [0], [1], or nothing (meaning both)
optimiser = ['Adam', 'RMSprop', 'SGD'][:1] #Use : to keep list type
learning_rate = [1e-1, 1e-2, 1e-3][:1] #e.g. [1:2] for second element
regu = [0, 5e-3, 5e-2, 5e-1][:1] #or [-1:] for last element

noise_avg, noise_std = 0, 1


###############################################
################ HELPER FUNCTIONS #############
###############################################

def error(data):
    mean = np.mean(data)
    standard_error = np.std(data, ddof=1) / np.sqrt(len(data))
    return mean, standard_error

def process_data(data):
    data_len = len(data[0])
    data_mean = np.zeros(data_len)
    data_error = np.zeros(data_len)
    for i in range(data_len):
        data_mean[i], data_error[i] = error(data[:, i])
    return data_mean, data_error

###############################################
################# LOAD DATA ###################
###############################################

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): device = "cuda"; num_epochs = num_epochs_cuda; print(f"CUDA is available. Setting epochs to {num_epochs}.")
else: device = "cpu"; num_epochs = num_epochs_no_cuda; print(f"CUDA is not available. Setting epochs to {num_epochs}.")

print("Loading MNIST data...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Collect test and train images and labels for evaluation
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

train_loader = []; val_loader = []
dataset_sizes = [int(len(train_images) * perc) for perc in dataset_portions]
for subset_size in dataset_sizes:
    subset_indices = np.random.choice(len(train_data), subset_size, replace=False)
    subset_train_data = Subset(train_data, subset_indices)

    # Split training data into training and validation sets
    train_indices, val_indices = train_test_split(np.arange(len(subset_train_data)), test_size=0.1, random_state=42)
    train_subset = Subset(subset_train_data, train_indices)
    val_subset = Subset(subset_train_data, val_indices)

    train_loader.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
    val_loader.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))
    
###############################################
############# PRINT INFORMATION ###############
###############################################   
    
print("\n")
print("Dataset sizes: ", dataset_sizes)
print("Model(s): ", CNNmodel)
print("Optimiser(s): ", optimiser)
print("Learning rate(s): ", learning_rate)
print("Regularisation parameter(s): ", regu)
print("Number of experiments: ", num_experiments)
print("Noise average: ", noise_avg, " and noise standard deviation: ", noise_std)
print("\n")



#%%
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


#Selection of model
if CNNmodel == "scatterCNN":
    models = {"scatterCNN": {"model": ScatteringModel(J, L, order, input_shape=(1, 28, 28), device=device)}}
elif CNNmodel == "normalCNN":
    models = {"normalCNN": {"model": NormalModel(input_shape=(1, 28, 28))}}
elif CNNmodel == ["scatterCNN", "normalCNN"]:
    models = {
    "scatterCNN": {"model": ScatteringModel(J, L, order, input_shape=(1, 28, 28), device=device)},
    "normalCNN": {"model": NormalModel(input_shape=(1, 28, 28))}
    }
else:
    ("Model not found error. Please selected one of 'scatterCNN' and 'normalCNN'") 


###############################################
############  HELPER FUNCTIONS ################
###############################################

def reset_weights(m):
    '''
    This function will reset model weights to a specified initialization.
    Works for most common types of layers.
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    
def choose_optimizer(optimizer_name, model, learning_rate):
    '''
    This function selects the optimizer based on the optimizer name.
    '''
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print("Optimizer not found. Please select one of 'Adam', 'RMSprop', 'SGD'.")
    
def initialize_history(history, model_name, dataset_sizes, optimiser, learning_rate, regu, num_experiments):
    if model_name not in history:
        history[model_name] = {}
    for subset_size in dataset_sizes:
        for opt in optimiser:
            for lr in learning_rate:
                for r in regu:
                    config_key = "{}_{}_{}_{}".format(subset_size, opt, lr, r)
                    loss_key = "{}_loss".format(config_key)
                    val_loss_key = "{}_val_loss".format(config_key)
                    if loss_key not in history[model_name]:
                        history[model_name][loss_key] = [[] for _ in range(num_experiments)]
                    if val_loss_key not in history[model_name]:
                        history[model_name][val_loss_key] = [[] for _ in range(num_experiments)]

def initialize_accuracies(accuracies, model_name, dataset_sizes, optimiser, learning_rate, regu, num_experiments):
    if model_name not in accuracies:
        accuracies[model_name] = {}
    for subset_size in dataset_sizes:
        for opt in optimiser:
            for lr in learning_rate:
                for r in regu:
                    config_key = "{}_{}_{}_{}".format(subset_size, opt, lr, r)
                    acc_key = "{}_accuracy".format(config_key)
                    if acc_key not in accuracies[model_name]:
                        accuracies[model_name][acc_key] = [[] for _ in range(num_experiments)]

                    
###############################################
################ FIT MODEL ####################
###############################################

        
criterion = nn.CrossEntropyLoss()

# Initialize accuracies using your model and subset size variables
history = {model_name: {} for model_name in models.keys()}
accuracies = {model_name: {} for model_name in models.keys()}

i=0

# Loop over each model for training
for model_name, model_details in models.items():
    print(f"Training {model_name} model...")
    model = model_details["model"].to(device)
    initialize_history(history, model_name, dataset_sizes, optimiser, learning_rate, regu, num_experiments)
    initialize_accuracies(accuracies, model_name, dataset_sizes, optimiser, learning_rate, regu, num_experiments)
    
    for lr in learning_rate:
        for opt in optimiser:
            optimizer = choose_optimizer(opt, model, lr)
                
            for reg in regu:
                print("Training with optimizer: ", opt, " and learning rate: ", lr, " and regularisation: ", reg)

                for subset_size in dataset_sizes:
                    config_key = "{}_{}_{}_{}".format(subset_size, opt, lr, reg)
                    acc_key = "{}_accuracy".format(config_key)
                    loss_key = "{}_loss".format(config_key)
                    val_loss_key = "{}_val_loss".format(config_key)

                    for experiment in range(num_experiments):
                        model.apply(reset_weights)

                        for epoch in tqdm(range(num_epochs), desc=f'Training with dataset size {subset_size}'):
                            model.train()
                            total_loss = 0
                            
                            for images, labels in train_loader[experiment]:
                                images, labels = images.to(device), labels.to(device)
                                optimizer.zero_grad()
                                outputs = model(images).to(device)
                                # print(model.classifier.parameters())
                                abs_weight_matrix = torch.abs(list(model.classifier.parameters())[2].data.detach()).to(device)
                                l1_loss = reg * torch.sum(abs_weight_matrix).to(device)
                                # out = list(model.classifier.parameters())[3].data
                                # print(l1_loss)
                                # print(out)
                                loss = criterion(outputs, labels) + l1_loss
                                # loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()
                                total_loss += loss.item() * images.size(0)

                            # Correctly update the history for the current model and experiment
                            history[model_name][loss_key][experiment].append(total_loss / len(train_subset))
                            
                            model.eval()
                            val_loss = 0
                            correct = 0
                            total = 0
                            with torch.no_grad():
                                for images, labels in val_loader[experiment]:
                                    # noise = torch.randn_like(images) * math.sqrt(1 / 10)
                                    shape = images.shape
                                    noise = np.random.normal(loc=noise_avg, scale=noise_std, size=shape)
                                    noisy_images = images + noise
                                    images = images.to(device)
                                    labels = labels.to(device)
                                    noisy_images = noisy_images.to(device).float()
                                    # outputs = model(images).to(device)
                                    outputs = model(noisy_images).to(device).float()
                                    loss = criterion(outputs, labels)
                                    val_loss += loss.item() * images.size(0)
                                    _, predicted = torch.max(outputs.data, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()

                            history[model_name][val_loss_key][experiment].append(val_loss / len(val_subset))                      
                            accuracy = correct / total
                            accuracies[model_name][acc_key][experiment].append(accuracy)

# %%

###############################################
############ EVALUATION FUNCTION ##############
###############################################

def display_examples(X, y_true, y_pred, indices, title):
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(indices[:3]):  # Display first three examples
        plt.subplot(1, 3, i + 1)
        plt.imshow(X[idx].squeeze(), cmap='gray',
                   interpolation='none')  # Make sure to squeeze in case there's an extra singleton dimension
        plt.title(f"{title}\nTrue: {y_true[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, model_name, test_images_tensor, test_labels):
    outputs = model(test_images_tensor)
    _, pred_labels = torch.max(outputs, 1)
    pred_labels = pred_labels.cpu().numpy()

    # Calculate metric values
    accuracy = accuracy_score(test_labels, pred_labels)
    precision = precision_score(test_labels, pred_labels, average='macro')
    recall = recall_score(test_labels, pred_labels, average='macro')
    f1 = f1_score(test_labels, pred_labels, average='macro')
    auc = roc_auc_score(test_labels, torch.softmax(outputs, dim=1).cpu().detach().numpy(), multi_class="ovr")

    print(f"Model: {model_name}")
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
    plt.title(f'Normalized Confusion Matrix for {model_name}')
    plt.show()

    # Display examples of True and False Predictions
    successful_indices = [i for i, (true, pred) in enumerate(zip(test_labels, pred_labels)) if true == pred]
    unsuccessful_indices = [i for i, (true, pred) in enumerate(zip(test_labels, pred_labels)) if true != pred]

    display_examples(np.array(test_images), test_labels, pred_labels, successful_indices, f"Successful Predictions for {model_name}")
    display_examples(np.array(test_images), test_labels, pred_labels, unsuccessful_indices, f"Unsuccessful Predictions for {model_name}")


#%%
###############################################
############ PRINT EVALUATIONS ################
###############################################

# Ensure test_images is a tensor and on the correct device
test_images_tensor = torch.stack(test_images).to(device)

# Evaluate each model separately
for model_name, model_details in models.items():
    model = model_details["model"].to(device)
    evaluate_model(model, model_name, test_images_tensor, np.array(test_labels))
  # Plot training & validation loss values for each dataset size
  
  
# When processing or plotting data, use the full configuration key
for model_name, model_details in models.items():
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)

    for opt in optimiser:
        for lr in learning_rate:
            for reg in regu:
                for subset_size in dataset_sizes:
                    config_key = "{}_{}_{}_{}".format(subset_size, opt, lr, reg)
                    loss_key = "{}_loss".format(config_key)
                    val_loss_key = "{}_val_loss".format(config_key)
                    print("Loss key: ", loss_key)

                    if loss_key in history[model_name]:
                        loss_data = np.array(history[model_name][loss_key])
                        val_loss_data = np.array(history[model_name][val_loss_key])

                        loss_mean, loss_error = process_data(loss_data)
                        val_loss_mean, val_loss_error = process_data(val_loss_data)

                        # Plot train loss with solid line
                        plt.errorbar(epochs, loss_mean, yerr=loss_error, fmt='-', label=f'{model_name} {opt} {lr} {reg} Train Loss')

                        # Plot validation loss with dashed line
                        plt.errorbar(epochs, val_loss_mean, yerr=val_loss_error, fmt='--', label=f'{model_name} {opt} {lr} {reg} Val Loss')

    plt.title(f'Model loss for all configurations')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.legend()
    plt.show()


plt.title('Validation Accuracy vs Dataset Size')
plt.xlabel('Training dataset Size')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
