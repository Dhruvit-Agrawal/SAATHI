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
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['model']['learning_rate'])
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

    def _validate_epoch(self) -> tuple[float, float, dict]:
        """Runs a single validation epoch and calculates detailed metrics."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)

                # Collect predictions and labels for detailed metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate basic metrics
        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

        # Calculate detailed metrics
        detailed_metrics = self._calculate_detailed_metrics(all_labels, all_preds)

        return val_loss, val_acc, detailed_metrics

    def _calculate_detailed_metrics(self, true_labels, pred_labels):
        """Calculate detailed evaluation metrics."""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = sum(np.array(pred_labels) == np.array(true_labels)) / len(true_labels)

        # Multi-class metrics with different averaging strategies
        metrics['precision_macro'] = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

        metrics['precision_micro'] = precision_score(true_labels, pred_labels, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(true_labels, pred_labels, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(true_labels, pred_labels, average='micro', zero_division=0)

        metrics['precision_weighted'] = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

        return metrics

    def _save_confusion_matrix(self, true_labels, pred_labels, epoch, class_names=None):
        """Save confusion matrix as artifact."""
        try:
            cm = confusion_matrix(true_labels, pred_labels)

            plt.figure(figsize=(10, 8))
            if class_names:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
            else:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

            plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Save temporarily
            cm_path = f"confusion_matrix_epoch_{epoch + 1}.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Log to MLflow
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")

            # Clean up
            if os.path.exists(cm_path):
                os.remove(cm_path)

        except Exception as e:
            print(f"Warning: Could not save confusion matrix: {e}")

    def train(self):
        """Executes the full training process with MLflow integration."""
        print("Starting model training with MLflow logging...")
        best_val_acc = 0.0
        best_model_state = None
        best_metrics = None

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
                val_loss, val_acc, detailed_metrics = self._validate_epoch()
                end_time = time.time()
                epoch_duration = end_time - start_time

                print(f"Epoch {epoch + 1}/{self.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"F1 (macro): {detailed_metrics['f1_macro']:.4f} | "
                      f"Duration: {epoch_duration:.2f}s")

                # Log metrics to MLflow - IMPORTANT: Pass actual calculated values, not functions
                metrics_to_log = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch_duration": epoch_duration
                }

                # Add detailed metrics
                metrics_to_log.update(detailed_metrics)

                # Log all metrics
                mlflow.log_metrics(metrics_to_log, step=epoch)

                # Check for the best model and save its state
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    best_metrics = detailed_metrics.copy()
                    print(f"-> New best validation accuracy: {best_val_acc:.4f} at epoch {epoch + 1}")

                    # Save confusion matrix for best epoch
                    # You'll need to modify this if you have class names
                    if epoch == self.epochs - 1 or val_acc > best_val_acc:  # Save on last epoch or best performance
                        all_preds, all_labels = self._get_all_predictions()
                        self._save_confusion_matrix(all_labels, all_preds, epoch)

            # --- 3. Log Final Best Metrics ---
            if best_metrics:
                print(f"\nBest Model Performance:")
                for metric, value in best_metrics.items():
                    print(f"{metric}: {value:.4f}")
                    # Log best metrics with prefix
                    mlflow.log_metric(f"best_{metric}", value)

            # --- 4. Save Final Best Model ---
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
                local_path = self.config.get('paths', {}).get('local_model_save_path')
                if local_path:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    torch.save(self.model.state_dict(), local_path)
                    print(f"[INFO] Best model also saved locally to: {local_path}")
            else:
                print("[WARNING] No best model was found to save.")

        # The MLflow run is now complete and has been saved.
        print("\nFinished Training!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")

    def _get_all_predictions(self):
        """Get all predictions for confusion matrix generation."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_preds, all_labels