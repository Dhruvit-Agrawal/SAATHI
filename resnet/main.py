# main.py

import yaml
import mlflow
import logging
from data_manager import DataManager
from model import create_model
from trainer import Trainer
import torch

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """Main function to run the training pipeline."""
    # 1. Load Configuration
    logging.info("Loading configuration from config.yaml...")
    with open(r'resnet/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup MLflow
    if config['use_mlflow']:
        logging.info(f"Setting up MLflow experiment '{config['experiment_name']}'...")
        mlflow.set_tracking_uri(config['tracking_uri'])
        mlflow.set_experiment(config['experiment_name'])

    # 3. Prepare Data
    logging.info("Preparing dataloaders...")
    data_manager = DataManager(config)
    train_loader, val_loader, test_loader, _, class_names = data_manager.prepare_dataloaders()

    # 4. Create Model
    num_classes = len(class_names)
    logging.info(f"Creating model for {num_classes} classes...")
    model = create_model(num_classes=num_classes)

    # 5. Start Training
    logging.info("Initializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

    # 6. Load the best saved model
    logging.info("Loading best model for final testing...")
    model.load_state_dict(torch.load(config['model_save_path']))

    # 7. Run final evaluation on the test set
    logging.info("Running final evaluation on the test set...")
    test_loss, test_acc, test_metrics = trainer._evaluate_epoch(test_loader) # Reusing the function!

    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print("Detailed Test Metrics:")
    print(test_metrics)

    #Log final test metrics to MLflow
    mlflow.log_metrics({
        "final_test_loss": test_loss,
        "final_test_accuracy": test_acc
    })

    logging.info("Pipeline finished successfully.")


if __name__ == '__main__':
    main()