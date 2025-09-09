# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import time
import os
from typing import Dict


class Trainer:
    """Handles the model training and validation loops with detailed MLflow logging."""

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Dict):
        """Initializes the Trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        # Assumes the model has a final fully connected layer named 'fc'
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.config['model']['learning_rate'])
        self.epochs = self.config['model']['epochs']

    def _train_epoch(self) -> float:
        """Runs a single training epoch."""
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        return running_loss / len(self.train_loader.dataset)

    def _validate_epoch(self) -> tuple[float, float]:
        """Runs a single validation epoch."""
        self.model.eval()
        running_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = corrects.double() / len(self.val_loader.dataset)
        return val_loss, val_acc.item()

    def train(self):
        """Executes the full training process with MLflow integration."""
        print("Starting model training with MLflow logging...")
        best_val_acc = 0.0
        best_model_state = None  # To keep track of the best model's state in memory

        # Set the MLflow experiment and get a sample input for the model signature
        mlflow.set_experiment(self.config['experiment_name'])
        input_example, _ = next(iter(self.val_loader))

        # All logging must happen inside this 'with' block
        with mlflow.start_run(run_name=self.config['run_name']) as run:
            print(f"MLflow Run Name: {self.config['run_name']}")
            print(f"MLflow Run ID: {run.info.run_id}")

            # --- 1. Log Initial Parameters, Tags, and Dataset ---
            mlflow.log_params(self.config['model'])
            mlflow.set_tags(self.config['tags'])
            mlflow.set_tag("model_name", self.config['model']['name'])
            mlflow.set_tag("mlflow.note.content", self.config['description'])

            dataset_path = self.config.get('paths', {}).get('dataset_path')
            if dataset_path and os.path.exists(dataset_path):
                mlflow.log_artifact(dataset_path, artifact_path="dataset")

            # --- 2. Training Loop ---
            for epoch in range(self.epochs):
                start_time = time.time()
                train_loss = self._train_epoch()
                val_loss, val_acc = self._validate_epoch()
                end_time = time.time()
                epoch_duration = end_time - start_time

                print(f"Epoch {epoch + 1}/{self.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Duration: {epoch_duration:.2f}s")

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                }, step=epoch)

                # Check for the best model and save its state
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    print(f"-> New best validation accuracy: {best_val_acc:.4f} at epoch {epoch + 1}")

            # --- 3. Save Final Best Model (after loop, inside 'with') ---
            print("\n[INFO] Training loop finished. Logging the best model...")
            if best_model_state:
                self.model.load_state_dict(best_model_state)

                # Log to MLflow Artifacts
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    input_example=input_example.cpu().numpy(),
                    registered_model_name="best_model"
                )
                print("[INFO] Best model successfully logged to MLflow artifacts.")

                # Save Locally
                local_path = self.config.get('paths', {}).get('local_model_path')
                if local_path:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    torch.save(self.model.state_dict(), local_path)
                    print(f"[INFO] Best model also saved locally to: {local_path}")
            else:
                print("[WARNING] No best model was found to save.")

        # The MLflow run is now complete and has been saved.
        print("\nFinished Training!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")