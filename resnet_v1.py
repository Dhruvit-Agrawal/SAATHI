# 1. SETUP: defining the dataset and model path and other variables
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import mlflow
import mlflow.pytorch
import json
import numpy as np

# DEFINE YOUR PATHS AND EXPERIMENT NAME HERE
data_dir = '/Users/kirtansakariya/Desktop/minor_p/datasets/cotton_dataset' # Set this path
model_path = '/Users/kirtansakariya/Desktop/minor_p/model/resnet' # desired local model save path
mlflow.set_experiment("Image Classification ResNet18") # MLflow experiment name

# Hyperparameters
batch_size = 8
epochs = 10
learning_rate = 0.001

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For reproducibility
torch.manual_seed(42)

# Create directory to save the model if it doesn't exist
# Note: mlflow will also save the model in its own artifact store
if model_path:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)


# 2. DATA PREPARATION: transforms, loading, and splitting
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([ # Validation and Test transforms are the same
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the dataset using ImageFolder
try:
    full_dataset = datasets.ImageFolder(data_dir)
except FileNotFoundError:
    print(f"Error: Data directory not found at '{data_dir}'. Please set the correct path.")
    exit()


# Split the data into training (64%), validation (16%), and testing (20%)
train_val_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_val_size
train_val_dataset, test_dataset_raw = random_split(full_dataset, [train_val_size, test_size])

train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset_raw, val_dataset_raw = random_split(train_val_dataset, [train_size, val_size])

print(f"Training size: {len(train_dataset_raw)}")
print(f"Validation size: {len(val_dataset_raw)}")
print(f"Test size: {len(test_dataset_raw)}")


# A simple wrapper to apply the correct transform to the subsets from random_split
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y
    def __len__(self):
        return len(self.subset)

train_dataset = TransformedDataset(train_dataset_raw, data_transforms['train'])
val_dataset = TransformedDataset(val_dataset_raw, data_transforms['val_test'])
test_dataset = TransformedDataset(test_dataset_raw, data_transforms['val_test'])


# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get class names and number of classes
class_names = full_dataset.classes
num_classes = len(class_names)
print("Class to Index mapping:", full_dataset.class_to_idx)
print(f"Number of classes: {num_classes}")


# 3. MODEL DEFINITION
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)


# 4. MLFLOW RUN & TRAINING LOOP
with mlflow.start_run() as run:
    print(f"MLflow Run ID: {run.info.run_id}")

    # Log hyperparameters to MLflow
    mlflow.log_param("model_architecture", "resnet18")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    # Log the class mapping as a JSON artifact
    with open("class_to_idx.json", "w") as f:
        json.dump(full_dataset.class_to_idx, f)
    mlflow.log_artifact("class_to_idx.json")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)

        # Print and log epoch statistics
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_val_acc.item(), step=epoch)

    print("\nFinished Training!")


    # 5. TESTING LOOP
    print("\nEvaluating on final Test Set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # Log the final test accuracy
    mlflow.log_metric("final_test_accuracy", test_accuracy)


    # 6. SAVING THE MODEL
    # Save locally
    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved locally to: {model_path}")

    # Log the model to MLflow
    mlflow.pytorch.log_model(model, "model", registered_model_name="ResNet18_Classifier")
    print("Model logged to MLflow.")


