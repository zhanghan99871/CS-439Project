import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, datasets
from torchsummary import summary
from kymatio.torch import Scattering2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
torch.manual_seed(10086)
torch.cuda.manual_seed(10086)
np.random.seed(10086)
torch.cuda.manual_seed_all(10086)
torch.backends.cudnn.deterministic = True

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################
################ CONFIGURATION ################
###############################################

dataset_portions = [1e-3, 5e-3, 2e-2, 1e-1, 1]     # Portions of complete dataset for the accuracy vs dataset size
num_experiments = 5                             # Number of experiments to run
num_epochs_cuda, num_epochs_no_cuda = 50, 20    # How many epochs to run
batch_size = 256                                # For reading in data

J, L, order = 3, 8, 2                           # Optimal values = log2(img_shape[1])-3,6-8; mopt=2
CNNmodel = ["scatterCNN", "normalCNN"][0]       # Pick [0], [1], or nothing (meaning both)
optimizer_list = ['Adam', 'RMSprop', 'SGD'][0:1] # Use : to keep list type
learning_rate = [1e-1, 1e-2, 1e-3][1:2]        # e.g. [1:2] for second element
regu = [0, 5e-4, 5e-3, 5e-2, 5e-1]             # or [-1:] for last element

noise_avg, noise_std = 0, 1

file = open("log.txt", 'w')
plotsummary = True

# %% [markdown]
# ### Prepatory steps
if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_no_cuda

#################################################
####################### LOAD DATA ###############
#################################################

if torch.cuda.is_available():
    device = "cuda"
    num_epochs = num_epochs_cuda
    print(f"CUDA is available. Setting epochs to {num_epochs}.")
else:
    device = "cpu"
    num_epochs = num_epochs_no_cuda
    print(f"CUDA is not available. Setting epochs to {num_epochs}.")

print("Loading MNIST data...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
# Collect test and train images and labels for evaluation
test_images = []
test_labels = []
for images, labels in test_loader:
    test_images.append(images)
    test_labels.append(labels)

train_images = []
train_labels = []
for images, labels in train_loader:
    train_images.append(images)
    train_labels.append(labels)

# Convert lists of tensors to tensors
test_images = torch.cat(test_images).to(DEVICE)
test_labels = torch.cat(test_labels).to(DEVICE)
train_images = torch.cat(train_images).to(DEVICE)
train_labels = torch.cat(train_labels).to(DEVICE)

print("shape of train_images: ", train_images.shape)

start_time = time.time()

if CNNmodel == "scatterCNN":
    scattering = Scattering2D(J=J, L=L, max_order=order, shape=(28, 28)).to(DEVICE)
    train_images_scat_coeffs = scattering(train_images).view(train_images.size(0), -1).to(DEVICE)
    print("train_images_scat_coeffs.shape: ", train_images_scat_coeffs.shape)
    test_images_scat_coeffs = scattering(test_images).view(test_images.size(0), -1).to(DEVICE)
    train_dataset = TensorDataset(train_images_scat_coeffs, train_labels)
    test_dataset = TensorDataset(test_images_scat_coeffs, test_labels)
else:
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
# Create DataLoader for the subset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to compute scattering coefficients: {elapsed_time:.2f} seconds")

dataset_sizes = [int(len(train_images) * perc) for perc in dataset_portions]


###############################################
############# PRINT INFORMATION ###############
###############################################

file.write("\n")
file.write("Dataset sizes: {}\n".format(dataset_sizes))
file.write("Model(s): {}\n".format(CNNmodel))
file.write("Optimiser(s): {}\n".format(optimizer_list))
file.write("Learning rate(s): {}\n".format(learning_rate))
file.write("Regularisation parameter(s): {}\n".format(regu))
file.write("Number of experiments: {}\n".format(num_experiments))
file.write("Noise average: {} and noise standard deviation: {}\n".format(noise_avg, noise_std))
file.write("\n")


###############################################
############# DEFINE MODEL ####################
###############################################

class ScatteringModel(nn.Module):
    def __init__(self, input_size=1953, num_classes=10):
        super(ScatteringModel, self).__init__()

        # Define fully connected layers with batch normalization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 32),  # Reduce the number of parameters by decreasing the number of units
            nn.BatchNorm1d(32),  # Add batch normalization
            nn.ReLU(),
            nn.Linear(32, 64),  # Reduce the number of parameters by decreasing the number of units
            nn.BatchNorm1d(64),  # Add batch normalization
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output for the number of classes
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

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

if CNNmodel == "scatterCNN":
    dummy_input = torch.zeros(1, 1, 28, 28).to(DEVICE)
    scattering = Scattering2D(J=J, L=L, max_order=order, shape=(28, 28)).to(DEVICE)
    dummy_scattering_output = scattering(dummy_input)
    input_size = dummy_scattering_output.view(dummy_scattering_output.size(0), -1).size(1)
    models = {"scatterCNN": {"model": ScatteringModel(input_size=input_size)}}
elif CNNmodel == "normalCNN":
    models = {"normalCNN": {"model": NormalModel(input_shape=(1, 28, 28))}}
elif CNNmodel == ["scatterCNN", "normalCNN"]:
    models = {
        "scatterCNN": {"model": ScatteringModel(input_size=1953,)},
        "normalCNN": {"model": NormalModel(input_shape=(1, 28, 28))}
    }
else:
    raise ValueError("Model not found error. Please select one of 'scatterCNN' or 'normalCNN'")

# Apply summary to each model individually
if plotsummary:
    for model_name, model_details in models.items():
        if model_name == "scatterCNN":
            summary(model_details["model"], input_size=(1953,), device=DEVICE)
        else:
            summary(model_details["model"], input_size=(1, 28, 28), device=DEVICE)


###############################################
############## HELPER FUNCTIONS ###############
###############################################


def error(data):
    mean = np.mean(data)
    if len(data) > 1:
        standard_error = np.std(data, ddof=1) / np.sqrt(len(data))
    else:
        standard_error = 0
    return mean, standard_error

def matrix_square_sum(matrix):
    return sum([pow(num, 2) for row in matrix for num in row])

def process_data(data, type="Loss"):
    data = np.array(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    if type == "Loss":
        data_mean = np.mean(data, axis=-2)  # Average over num_experiments
        if data.shape[-2] > 1:
            data_std = np.std(data, axis=-2, ddof=1) / np.sqrt(data.shape[-2])
        else:
            data_std = np.zeros_like(data_mean)
        return data_mean, data_std
    elif type == "Accuracy":
        data_mean_vec = np.mean(data, axis=-2)
        data_mean = np.max(data_mean_vec)  # Average over num_experiments
        maxind = np.where(data_mean == data_mean_vec)[0]
        if maxind.size > 1:
            maxind = maxind[0]
        if data.shape[-2] > 1:
            data_std_vec = np.std(data, axis=-2, ddof=1) / np.sqrt(data.shape[-2])
            data_std = data_std_vec[maxind]
        else:
            data_std = np.zeros_like(data_mean)
        return data_mean, data_std

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


def initialize_history(history, model_name, dataset_sizes, optimizer, learning_rate, regu, num_experiments):
    if model_name not in history:
        history[model_name] = {}
    for subset_size in dataset_sizes:
        for opt in optimizer:
            for lr in learning_rate:
                for r in regu:
                    config_key = "{}_{}_{}_{}".format(subset_size, opt, lr, r)
                    loss_key = "{}_loss".format(config_key)
                    val_loss_key = "{}_val_loss".format(config_key)
                    if loss_key not in history[model_name]:
                        history[model_name][loss_key] = [[] for _ in range(num_experiments)]
                    if val_loss_key not in history[model_name]:
                        history[model_name][val_loss_key] = [[] for _ in range(num_experiments)]


def initialize_accuracies(accuracies, model_name, dataset_sizes, optimizer, learning_rate, regu, num_experiments):
    if model_name not in accuracies:
        accuracies[model_name] = {}
    for subset_size in dataset_sizes:
        for opt in optimizer:
            for lr in learning_rate:
                for r in regu:
                    config_key = "{}_{}_{}_{}".format(subset_size, opt, lr, r)
                    acc_key = "{}_accuracy".format(config_key)
                    if acc_key not in accuracies[model_name]:
                        accuracies[model_name][acc_key] = [[] for _ in range(num_experiments)]
                        
def display_examples(X, y_true, y_pred, indices, title):
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(indices[:3]):  # Display first three examples
        plt.subplot(1, 3, i + 1)
        plt.imshow(X[idx].squeeze(), cmap='gray',
                   interpolation='none')  # Make sure to squeeze in case there's an extra singleton dimension
        plt.title(f"{title}\nTrue: {y_true[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    plt.tight_layout()
    

###############################################
################ FIT MODEL ####################
###############################################


criterion = nn.CrossEntropyLoss()
# Initialize accuracies using your model and subset size variables
history = {model_name: {} for model_name in models.keys()}
accuracies = {model_name: {} for model_name in models.keys()}
precisions = {model_name: {} for model_name in models.keys()}
recalls = {model_name: {} for model_name in models.keys()}
f1_scores = {model_name: {} for model_name in models.keys()}
auc_scores = {model_name: {} for model_name in models.keys()}
# Loop over each model for training
for model_name, model_details in models.items():
    print(f"Training {model_name} model...")
    model = model_details["model"].to(device)
    initialize_history(history, model_name, dataset_sizes, optimizer_list, learning_rate, regu, num_experiments)
    initialize_accuracies(accuracies, model_name, dataset_sizes, optimizer_list, learning_rate, regu, num_experiments)

    for lr in learning_rate:
        for opt_name in optimizer_list:
            for reg in regu:
                print("Training with optimizer: ", opt_name, " and learning rate: ", lr, " and regularisation: ", reg)

                # Create the optimizer
                optimizer = choose_optimizer(opt_name, model, lr)

                for experiment in range(num_experiments):
                    for subset_size in dataset_sizes:
                        model.apply(reset_weights)
                        config_key = "{}_{}_{}_{}".format(subset_size, opt_name, lr, reg)
                        acc_key = "{}_accuracy".format(config_key)
                        loss_key = "{}_loss".format(config_key)
                        val_loss_key = "{}_val_loss".format(config_key)
                        prec_key = "{}_precision".format(config_key)
                        rec_key = "{}_recall".format(config_key)
                        f1_key = "{}_f1".format(config_key)
                        auc_key = "{}_auc".format(config_key)
                      
                        
                        # Create DataLoader for the subset
                        subset_indices = list(range(subset_size))
                        subset_train_dataset = Subset(train_dataset, subset_indices)
                        subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

                        for epoch in tqdm(range(num_epochs), desc=f'Training with dataset size {subset_size}'):
                            model.train()
                            total_loss = 0
                            total_images = 0

                            for images, labels in subset_train_loader:
                                images, labels = images.to(device), labels.to(device)
                                optimizer.zero_grad()
                                outputs = model(images)
                                
                                # Calculate L1 and L2 regularization only for Linear layers
                                l1_loss = 0
                                l2_loss = 0
                                for layer in model.classifier:
                                    if isinstance(layer, nn.Linear):
                                        l1_loss += reg * torch.sum(torch.abs(layer.weight))
                                        l2_loss += reg * torch.sum(layer.weight ** 2)

                                loss = criterion(outputs, labels) + l2_loss

                                loss.backward()
                                optimizer.step()
                                total_loss += loss.item() * images.size(0)
                                total_images += images.size(0)

                            average_loss = total_loss / total_images
                            history[model_name][loss_key][experiment].append(average_loss)

                            model.eval()
                            val_total_loss = 0
                            val_total_images = 0
                            correct = 0

                            with torch.no_grad():
                                for images, labels in test_loader:
                                    images, labels = images.to(device), labels.to(device)
                                    outputs = model(images)
                                    loss = criterion(outputs, labels)
                                    val_total_loss += loss.item() * images.size(0)
                                    val_total_images += images.size(0)
                                    _, predicted = torch.max(outputs.data, 1)
                                    correct += (predicted == labels).sum().item()

                            val_average_loss = val_total_loss / val_total_images
                            history[model_name][val_loss_key][experiment].append(val_average_loss)
                            accuracy = correct / val_total_images

                        if model_name == "scatterCNN":
                            outputs = model(test_images_scat_coeffs.to(DEVICE))
                        else:
                            outputs = model(test_images.to(DEVICE))
                        _, pred_labels = torch.max(outputs, 1)
                        pred_labels = pred_labels.cpu().numpy()

                        # Calculate metric values
                        accuracy = accuracy_score(test_labels.cpu(), pred_labels)
                        precision = precision_score(test_labels.cpu(), pred_labels, average='macro')
                        recall = recall_score(test_labels.cpu(), pred_labels, average='macro')
                        f1 = f1_score(test_labels.cpu(), pred_labels, average='macro')

                        # Apply softmax to the model outputs before computing ROC AUC
                        probabilities = torch.softmax(outputs, dim=1).cpu().detach().numpy()
                        auc_value = roc_auc_score(test_labels.cpu(), probabilities, multi_class="ovr")

                        accuracies[model_name][acc_key][experiment] = [accuracy]
                        precisions[model_name][prec_key] = [precision]
                        recalls[model_name][rec_key] = [recall]
                        f1_scores[model_name][f1_key] = [f1]
                        auc_scores[model_name][auc_key] = [auc_value]

                        file.write("Subset size: {}\n".format(subset_size))
                        file.write("Model: {}\n".format(model_name))
                        file.write("optimizer: {}\n".format(opt_name))
                        file.write("learning rate: {}\n".format(lr))
                        file.write("regularization: {}\n".format(reg))
                        file.write("Test Accuracy: {}\n".format(accuracy))
                        file.write("Precision: {}\n".format(precision))
                        file.write("Recall: {}\n".format(recall))
                        file.write("F1 Score: {}\n".format(f1))
                        file.write("AUC: {}\n".format(auc_value))
                        file.write("\n")

                        # Plot subset-size-dependent l2_reg
                        config_key = "{}_{}_{}_{}".format(subset_size, opt_name, lr, reg)

                        # Calculate ROC curve
                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        for i in range(10):  # Assuming 10 classes for MNIST
                            fpr[i], tpr[i], _ = roc_curve(np.array(test_labels.cpu()) == i, probabilities[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])

                        # Plot ROC curve
                        plt.figure()
                        for i in range(10):
                            plt.plot(fpr[i], tpr[i], lw=2, label='Digit {} (AUC = {:.2f})'.format(i, roc_auc[i]))
                        plt.plot([0, 1], [0, 1], 'k--', lw=2)
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.0])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'ROC Curve for {model_name} with {opt_name} optimizer, \n learning rate = {lr}, regularisation parameter = {reg}, and {subset_size} samples')
                        plt.legend(loc="lower right")
                        plt.text(0.2, 0.1, 'Total AUC = {:.2f}'.format(auc_value), fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
                        plt.savefig('./l2_reg/{}_roc_curve.png'.format(config_key))
                        plt.close()

                        # Generate and plot the normalized confusion matrix
                        cm = confusion_matrix(test_labels.cpu(), pred_labels, normalize='true')
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt=".2%", linewidths=.5, square=True, cmap='Blues', ax=ax)
                        ax.set_ylabel('True label')
                        ax.set_xlabel('Predicted label')
                        ax.set_title(f'Normalized Confusion Matrix for {model_name} with {opt_name} optimizer, \n learning rate = {lr}, regularisation parameter = {reg},'
                                     f'and {subset_size} samples and accuracy = {accuracy}')
                        plt.savefig('./l2_reg/{}_confusion_matrix.png'.format(config_key))
                        plt.close()

                        # Display examples of True and False Predictions
                        successful_indices = [i for i, (true, pred) in enumerate(zip(test_labels.cpu(), pred_labels)) if true == pred]
                        unsuccessful_indices = [i for i, (true, pred) in enumerate(zip(test_labels.cpu(), pred_labels)) if true != pred]

                        display_examples(np.array(test_images.cpu()), test_labels.cpu(), pred_labels, successful_indices,
                                         f"Successful Predictions for {model_name}")
                        plt.savefig('./l2_reg/{}_successful_examples.png'.format(config_key))
                        plt.close()

                        display_examples(np.array(test_images.cpu()), test_labels.cpu(), pred_labels, unsuccessful_indices,
                                         f"Unsuccessful Predictions for {model_name}")
                        plt.savefig('./l2_reg/{}_unsuccessful_examples.png'.format(config_key))
                        plt.close()


###############################################
############ PLOT ACCURACY AND LOSS ###########
###############################################

# Initialize figure and axes for subplots
fig, axs = plt.subplots(1, len(dataset_portions), figsize=(5 * len(dataset_portions), 5))  # Adjust figsize as needed
acc_mean_per_subset = np.zeros(len(dataset_sizes))
acc_std_per_subset = np.zeros(len(dataset_sizes))
prec_mean_per_subset = np.zeros(len(dataset_sizes))
prec_std_per_subset = np.zeros(len(dataset_sizes))
rec_mean_per_subset = np.zeros(len(dataset_sizes))
rec_std_per_subset = np.zeros(len(dataset_sizes))
f1_mean_per_subset = np.zeros(len(dataset_sizes))
f1_std_per_subset = np.zeros(len(dataset_sizes))
auc_mean_per_subset = np.zeros(len(dataset_sizes))
auc_std_per_subset = np.zeros(len(dataset_sizes))
# Iterate through each model in the models dictionary
plt.clf()
for model_name, model_details in models.items():
    epochs = range(1, num_epochs + 1)

    for opt in optimizer_list:
        for lr in learning_rate:
            for reg in regu:
                for subset_size in dataset_sizes:
                    config_key = "{}_{}_{}_{}".format(subset_size, opt, lr, reg)
                    loss_key = "{}_loss".format(config_key)
                    val_loss_key = "{}_val_loss".format(config_key)
                    acc_key = "{}_accuracy".format(config_key)
                    prec_key = "{}_precision".format(config_key)
                    rec_key = "{}_recall".format(config_key)
                    f1_key = "{}_f1".format(config_key)
                    auc_key = "{}_auc".format(config_key)

                    if loss_key in history[model_name] and acc_key in accuracies[model_name]:
                        loss_data = np.array(history[model_name][loss_key])
                        val_loss_data = np.array(history[model_name][val_loss_key])
                        acc_data = np.array(accuracies[model_name][acc_key])
                        prec_data = np.array(precisions[model_name][prec_key])
                        rec_data = np.array(recalls[model_name][rec_key])
                        f1_data = np.array(f1_scores[model_name][f1_key])
                        auc_data = np.array(auc_scores[model_name][auc_key])

                        if loss_data.size and val_loss_data.size and acc_data.size:
                            loss_mean, loss_error = process_data(loss_data, "Loss")
                            val_loss_mean, val_loss_error = process_data(val_loss_data, "Loss")
                            acc_mean, acc_error = process_data(acc_data, "Accuracy")
                            prec_mean, prec_error = process_data(prec_data, "Accuracy")
                            rec_mean, rec_error = process_data(rec_data, "Accuracy")
                            f1_mean, f1_error = process_data(f1_data, "Accuracy")
                            auc_mean, auc_error = process_data(auc_data, "Accuracy")

                            plt.errorbar(epochs, loss_mean, yerr=loss_error, fmt='-', label=f'{model_name} {opt} {lr} {reg} Train Loss')
                            plt.errorbar(epochs, val_loss_mean, yerr=val_loss_error, fmt='--', label=f'{model_name} {opt} {lr} {reg} Val Loss')
                            plt.title(f"Training size {subset_size}")
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss')
                            plt.xscale('log')
                            plt.yscale('log')
                            plt.legend()
                            plt.savefig("./l2_reg/{}_loss.png".format(config_key))
                            plt.clf()

                            acc_mean_per_subset[dataset_sizes.index(subset_size)] = acc_mean
                            acc_std_per_subset[dataset_sizes.index(subset_size)] = acc_error
                            prec_mean_per_subset[dataset_sizes.index(subset_size)] = prec_mean
                            prec_std_per_subset[dataset_sizes.index(subset_size)] = prec_error
                            rec_mean_per_subset[dataset_sizes.index(subset_size)] = rec_mean
                            rec_std_per_subset[dataset_sizes.index(subset_size)] = rec_error
                            f1_mean_per_subset[dataset_sizes.index(subset_size)] = f1_mean
                            f1_std_per_subset[dataset_sizes.index(subset_size)] = f1_error
                            auc_mean_per_subset[dataset_sizes.index(subset_size)] = auc_mean
                            auc_std_per_subset[dataset_sizes.index(subset_size)] = auc_error

                plt.errorbar(dataset_sizes, acc_mean_per_subset, yerr=acc_std_per_subset, fmt='o-', capsize=5, capthick=2, label=f'{model_name} {opt} {lr} {reg} Accuracy')
                plt.errorbar(dataset_sizes, prec_mean_per_subset, yerr=prec_std_per_subset, fmt='s-', capsize=5, capthick=2, label=f'{model_name} {opt} {lr} {reg} Precision')
                plt.errorbar(dataset_sizes, rec_mean_per_subset, yerr=rec_std_per_subset, fmt='^-', capsize=5, capthick=2, label=f'{model_name} {opt} {lr} {reg} Recall')
                plt.errorbar(dataset_sizes, f1_mean_per_subset, yerr=f1_std_per_subset, fmt='d-', capsize=5, capthick=2, label=f'{model_name} {opt} {lr} {reg} F1 Score')
                plt.errorbar(dataset_sizes, auc_mean_per_subset, yerr=auc_std_per_subset, fmt='p-', capsize=5, capthick=2, label=f'{model_name} {opt} {lr} {reg} AUC')
                plt.title(f'Accuracy, Precision, Recall, F1 Score, and AUC over Different Dataset Sizes for {model_name} with {opt} optimizer,\n learning rate = {lr}, regularisation parameter = {reg}', fontsize=10)
                plt.xlabel('Dataset Size')
                plt.xscale('log')
                plt.ylabel('Metric Value')
                plt.legend(fontsize=10)
                plt.savefig("./l2_reg/{}_accuracy_metrics.png".format(config_key))
                plt.clf()

plt.tight_layout()

file.close()
