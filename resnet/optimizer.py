# optimize.py

import torch
import yaml
from model import create_model  # We use the same model definition
import json


def main():
    """Loads the trained model and saves a quantized version for inference."""
    print("Starting model optimization (quantization)...")

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load class mapping to get num_classes
    with open(config['class_map_path'], 'r') as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    # 1. Load the trained floating-point model
    # We create a model instance first and then load the state dict.
    # We must set pretrained=False as we are loading our own weights.
    model_fp32 = create_model(num_classes=num_classes, pretrained=False)
    model_fp32.load_state_dict(torch.load(config['local_model_save_path'], map_location='cpu'))
    model_fp32.eval()  # Set to evaluation mode

    # 2. Apply Dynamic Quantization
    # This is a post-training quantization method suitable for LSTMs and Transformers,
    # and works well for CNNs on CPU.
    model_quantized = torch.quantization.quantize_dynamic(
        model_fp32,  # The model to be quantized
        {torch.nn.Linear},  # Specify which layer types to quantize
        dtype=torch.qint8  # The target data type for quantized weights
    )

    # 3. Save the quantized model
    torch.save(model_quantized.state_dict(), config['quantized_model_save_path'])

    print("\nOptimization complete!")
    print(f"Original model path: {config['model_save_path']}")
    print(f"Quantized model saved to: {config['quantized_model_save_path']}")


if __name__ == '__main__':
    main()