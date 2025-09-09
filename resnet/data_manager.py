# data_manager.py

import torch
import json
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, List

class DataManager:
    """Handles all data loading, transformations, and splitting."""

    def __init__(self, config: Dict):
        """
        Initializes the DataManager with configuration.

        Args:
            config (Dict): A dictionary containing data-related parameters.
        """
        self.config = config
        # --- UPDATED TO USE NESTED CONFIG VALUES ---
        self.data_path = config['paths']['dataset_path']
        self.image_size = config['dataset']['image_size']
        self.batch_size = config['model']['batch_size']
        self.num_workers = config['model']['num_workers']
        self.class_map_path = config['paths']['class_map_path']

    def _get_transforms(self) -> Dict[str, transforms.Compose]:
        """Defines the data transformations for training and validation/testing."""
        return {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val_test': transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def prepare_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, List[str]]:
        """
        Loads the dataset, splits it, applies transforms, and creates DataLoaders.

        Returns:
            Tuple containing train_loader, val_loader, test_loader, class_to_idx, and class_names.
        """
        full_dataset = datasets.ImageFolder(self.data_path)

        # Split data: 64% train, 16% val, 20% test
        train_val_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_val_size
        train_val_set, test_set = random_split(full_dataset, [train_val_size, test_size])

        train_size = int(0.8 * len(train_val_set))
        val_size = len(train_val_set) - train_size
        train_set, val_set = random_split(train_val_set, [train_size, val_size])

        transforms_map = self._get_transforms()

        # Apply transforms by wrapping the datasets
        train_set.dataset.transform = transforms_map['train']
        val_set.dataset.transform = transforms_map['val_test']
        test_set.dataset.transform = transforms_map['val_test']

        # Create DataLoaders
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        class_to_idx = full_dataset.class_to_idx
        class_names = full_dataset.classes

        # Ensure the output directory exists before saving the class map
        output_dir = os.path.dirname(self.class_map_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save class_to_idx mapping for inference
        with open(self.class_map_path, 'w') as f:
            json.dump(class_to_idx, f)

        print(f"Dataset split: Train={len(train_set)}, Validation={len(val_set)}, Test={len(test_set)}")
        return train_loader, val_loader, test_loader, class_to_idx, class_names