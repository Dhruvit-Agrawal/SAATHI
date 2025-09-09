# model.py

from torchvision import models
import torch.nn as nn

def create_model(num_classes: int, pretrained: bool = True) -> models.ResNet:
    """
    Creates a ResNet-18 model with a modified classifier.

    Args:
        num_classes (int): The number of output classes.
        pretrained (bool): Whether to use pre-trained weights from ImageNet.

    Returns:
        models.ResNet: The configured ResNet-18 model.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze all the base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model