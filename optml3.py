import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchsummary import summary
from kymatio.torch import Scattering2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import warnings

torch.manual_seed(10086)
torch.cuda.manual_seed(10086)
np.random.seed(10086)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category = FutureWarning)

# %%
###############################################
################ CONFIGURATION ################
###############################################

dataset_portions = [0.0001, 0.0005, 0.001]  # Portions of complete dataset for the accuracy vs dataset size
#dataset_portions = [1e-4, 1e-3, 1e-2, 1e-1, 1]  # Portions of complete dataset for the accuracy vs dataset size
num_experiments = 1
num_epochs_cuda, num_epochs_no_cuda = 50, 50  # How many epochs to run
batch_size = 32  # For reading in data

J, L, order = 3, 8, 2  # Optimal values = log2(img_shape[1])-3,6-8; mopt=2
CNNmodel = ["scatterCNN", "normalCNN"][1]  # Pick [0], [1], or nothing (meaning both)
optimizer_list = ['Adam', 'RMSprop', 'SGD'][:1]  # Use : to keep list type
learning_rate = [1e-1, 1e-2, 1e-3][1:2]  # e.g. [1:2] for second element
regu = [0, 5e-3, 5e-2, 5e-1][:1]  # or [-1:] for last element

noise_avg, noise_std = 0, 1

file = open("log.txt", 'w')


###############################################
################ HELPER FUNCTIONS #############
###############################################

def error(data):
    mean = np.mean(data)
    standard_error = np.std(data, ddof=1) / np.sqrt(len(data))
    return mean, standard_error


def process_data(data, type="Loss"):
    if type == "Loss":
        data_mean = np.mean(data, axis=-2)  # Average over num_experiments
        data_std = np.std(data, axis=-2, ddof=1) / np.sqrt(data.shape[-2])
        return data_mean, data_std
    elif type == "Accuracy":
        data_mean_vec = np.mean(data, axis=-2)
        data_mean = np.max(data_mean_vec)  # Average over num_experiments
        maxind = np.where(data_mean == data_mean_vec)[0]
        if maxind.size > 1:
            maxind = maxind[0]
        data_std_vec = np.std(data, axis=-2, ddof=1) / np.sqrt(data.shape[-2])
        data_std = data_std_vec[maxind]
        return data_mean, data_std


###############################################
################# LOAD DATA ###################
###############################################

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = "cuda"; num_epochs = num_epochs_cuda; print(f"CUDA is available. Setting epochs to {num_epochs}.")
else:
    device = "cpu"; num_epochs = num_epochs_no_cuda; print(f"CUDA is not available. Setting epochs to {num_epochs}.")

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
    for i in range(len(images)):
        test_images.append(images[i])
        test_labels.append(labels[i])

train_images = []
train_labels = []
for images, labels in train_loader:
    for i in range(len(images)):
        train_images.append(images[i])
        train_labels.append(labels[i])

train_loader = []
val_loader = []
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

file.write("\n")
file.write("Dataset sizes: {}\n".format(dataset_sizes))
file.write("Model(s): {}\n".format(CNNmodel))
file.write("Optimiser(s): {}\n".format(optimizer_list))
file.write("Learning rate(s): {}\n".format(learning_rate))
file.write("Regularisation parameter(s): {}\n".format(regu))
file.write("Number of experiments: {}\n".format(num_experiments))
file.write("Noise average: {} and noise standard deviation: {}\n".format(noise_avg, noise_std))
file.write("\n")




# %%
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


# Selection of model
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
    

#summary(model=models, input_size=(1, 28, 28), device=device)


###############################################
############ HELPER FUNCTIONS #################
###############################################

def sreset_weights(m):
    '''
    This function will reset model weights to a specified initialization.
    Works for most common types of layers.
    '''
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        
        
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
test_images_tensor = torch.stack(test_images).to(device)
# Initialize accuracies using your model and subset size variables
history = {model_name: {} for model_name in models.keys()}
accuracies = {model_name: {} for model_name in models.keys()}

# Loop over each model for training
for model_name, model_details in models.items():
    print(f"Training {model_name} model...")
    model = model_details["model"].to(device)
    initialize_history(history, model_name, dataset_sizes, optimizer_list, learning_rate, regu, num_experiments)
    initialize_accuracies(accuracies, model_name, dataset_sizes, optimizer_list, learning_rate, regu, num_experiments)

    for lr in learning_rate:
        for opt in optimizer_list:
            optimizer = choose_optimizer(opt, model, lr)

            for reg in regu:
                print("Training with optimizer: ", opt, " and learning rate: ", lr, " and regularisation: ", reg)

                for experiment in range(num_experiments):
                    model.apply(reset_weights)

                    for subset_size in dataset_sizes:
                        config_key = "{}_{}_{}_{}".format(subset_size, opt, lr, reg)
                        acc_key = "{}_accuracy".format(config_key)
                        loss_key = "{}_loss".format(config_key)
                        val_loss_key = "{}_val_loss".format(config_key)

                        for epoch in tqdm(range(num_epochs), desc=f'Training with dataset size {subset_size}'):
                            model.train()
                            total_loss = 0
                            total_images = 0

                            for images, labels in train_loader[dataset_sizes.index(subset_size)]:
                                images, labels = images.to(device), labels.to(device)
                                optimizer.zero_grad()
                                outputs = model(images)
                                abs_weight_matrix = torch.abs(list(model.classifier.parameters())[2].data.detach())
                                l1_loss = reg * torch.sum(abs_weight_matrix)
                                loss = criterion(outputs, labels) + l1_loss
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
                                for images, labels in val_loader[dataset_sizes.index(subset_size)]:
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
                            accuracies[model_name][acc_key][experiment].append(accuracy)

                            # print(f'Epoch {epoch}:')
                            # print(f'Correct Predictions: {correct}')
                            # print(f'Total Samples: {total}')
                            # print(f'Accuracy this epoch: {accuracy}')

                        outputs = model(test_images_tensor)
                        _, pred_labels = torch.max(outputs, 1)
                        pred_labels = pred_labels.cpu().numpy()

                        # Calculate metric values
                        accuracy = accuracy_score(test_labels, pred_labels)
                        precision = precision_score(test_labels, pred_labels, average='macro')
                        recall = recall_score(test_labels, pred_labels, average='macro')
                        f1 = f1_score(test_labels, pred_labels, average='macro')

                        # Apply softmax to the model outputs before computing ROC AUC
                        probabilities = torch.softmax(outputs, dim=1).cpu().detach().numpy()
                        auc_value = roc_auc_score(test_labels, probabilities, multi_class="ovr")

                        file.write("Subset size: {}\n".format(subset_size))
                        file.write("Model: {}\n".format(model_name))
                        file.write("optimizer: {}\n".format(opt))
                        file.write("learning rate: {}\n".format(lr))
                        file.write("regularization: {}\n".format(reg))
                        file.write("Test Accuracy: {}\n".format(accuracy))
                        file.write("Precision: {}\n".format(precision))
                        file.write("Recall: {}\n".format(recall))
                        file.write("F1 Score: {}\n".format(f1))
                        file.write("AUC: {}\n".format(auc_value))
                        file.write("\n")
                        
                        #Plot subset-size-dependent figures
                        config_key = "{}_{}_{}_{}".format(subset_size, opt, lr, reg)

                        # Calculate ROC curve
                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        for i in range(10):  # Assuming 10 classes for MNIST
                            fpr[i], tpr[i], _ = roc_curve(np.array(test_labels) == i, probabilities[:, i])
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
                        plt.title('ROC Curve for {}'.format(model_name))
                        plt.legend(loc="lower right")
                        plt.text(0.2, 0.1, 'Total AUC = {:.2f}'.format(auc_value), fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
                        plt.savefig('./figures/{}_roc_curve.png'.format(config_key))
                        plt.close()
                        
                        # Generate and plot the normalized confusion matrix
                        cm = confusion_matrix(test_labels, pred_labels, normalize='true')
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt=".2%", linewidths=.5, square=True, cmap='Blues', ax=ax)
                        ax.set_ylabel('True label')
                        ax.set_xlabel('Predicted label')
                        ax.set_title(
                            f'Normalized Confusion Matrix for {model_name}\nwith {opt} optimizer, learning rate = {lr} , and regularisation parameter = {reg}')
                        plt.savefig('./figures/{}_confusion_matrix.png'.format(config_key))
                        plt.close()

                        # Display examples of True and False Predictions
                        successful_indices = [i for i, (true, pred) in enumerate(zip(test_labels, pred_labels)) if true == pred]
                        unsuccessful_indices = [i for i, (true, pred) in enumerate(zip(test_labels, pred_labels)) if true != pred]

                        display_examples(np.array(test_images), test_labels, pred_labels, successful_indices,
                                         f"Successful Predictions for {model_name}")
                        plt.savefig('./figures/{}_successful_examples.png'.format(config_key))
                        plt.close()

                        display_examples(np.array(test_images), test_labels, pred_labels, unsuccessful_indices,
                                         f"Unsuccessful Predictions for {model_name}")
                        plt.savefig('./figures/{}_unsuccessful_examples.png'.format(config_key))
                        plt.close()


###############################################
############ PLOT ACCURACY AND LOSS ###########
###############################################

# Initialize figure and axes for subplots
fig, axs = plt.subplots(1, len(dataset_portions), figsize=(5 * len(dataset_portions), 5))  # Adjust figsize as needed
acc_mean_per_subset = np.zeros(len(dataset_sizes))
acc_std_per_subset = np.zeros(len(dataset_sizes))
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

                    if loss_key in history[model_name] and acc_key in accuracies[model_name]:
                        loss_data = np.array(history[model_name][loss_key])
                        val_loss_data = np.array(history[model_name][val_loss_key])
                        acc_data = np.array(accuracies[model_name][acc_key])

                        if loss_data.size and val_loss_data.size and acc_data.size:
                            loss_mean, loss_error = process_data(loss_data, "Loss")
                            val_loss_mean, val_loss_error = process_data(val_loss_data, "Loss")

                            plt.errorbar(epochs, loss_mean, yerr=loss_error, fmt='-',
                                            label=f'{model_name} {opt} {lr} {reg} Train Loss')
                            plt.errorbar(epochs, val_loss_mean, yerr=val_loss_error, fmt='--',
                                            label=f'{model_name} {opt} {lr} {reg} Val Loss')
                            plt.title("Training size " + str(subset_size))
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss')
                            plt.xscale('log')
                            plt.yscale('log')
                            plt.legend()
                            plt.savefig("./figures/{}_loss.png".format(config_key))
                            plt.clf()

                            acc_mean, acc_error = process_data(acc_data, "Accuracy")

                            acc_mean_per_subset[dataset_sizes.index(subset_size)] = acc_mean
                            acc_std_per_subset[dataset_sizes.index(subset_size)] = acc_error


                plt.errorbar(dataset_sizes, acc_mean_per_subset, yerr=acc_std_per_subset, fmt='o-', capsize=5, capthick=2,
                            label=f'{model_name} {opt} {lr} {reg} Accuracy')
                plt.title('Accuracy over Different Dataset Sizes')
                plt.xlabel('Dataset Size')
                plt.xscale('log')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig("./figures/{}_accuracy.png".format(config_key))
                plt.clf()

plt.tight_layout()

file.close()
