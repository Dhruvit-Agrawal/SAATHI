# predictor.py

import torch
import json
from PIL import Image
from torchvision import transforms
import argparse
from model import create_model  # Reuse the model definition function


class ImagePredictor:
    """Handles loading a quantized model and making predictions on images."""

    def __init__(self, model_path: str, class_map_path: str):
        """
        Initializes the predictor.

        Args:
            model_path (str): Path to the quantized model state_dict.
            class_map_path (str): Path to the JSON file mapping class names to indices.
        """
        self.device = torch.device("cpu")

        # Load class mapping
        with open(class_map_path, 'r') as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        num_classes = len(self.idx_to_class)

        # Load the quantized model architecture and state
        self.model = self._load_quantized_model(model_path, num_classes)

        # Define the same transformation as validation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_quantized_model(self, model_path: str, num_classes: int) -> torch.nn.Module:
        """Loads the structure and state of a quantized model."""
        # Create a model instance
        model_fp32 = create_model(num_classes=num_classes, pretrained=False)

        # Prepare it for dynamic quantization
        model_quantized = torch.quantization.quantize_dynamic(
            model_fp32, {torch.nn.Linear}, dtype=torch.qint8
        )

        # Load the quantized state_dict
        model_quantized.load_state_dict(torch.load(model_path, map_location=self.device))
        model_quantized.eval()
        return model_quantized

    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Makes a prediction on a single image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            A tuple containing the predicted class name and the confidence score.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None, None

        # Preprocess the image
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.model(image_tensor)

            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(output, dim=1)

            # Get the top prediction
            confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = self.idx_to_class[predicted_idx.item()]
            confidence_score = confidence.item()

            return predicted_class, confidence_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict an image's class using a quantized model.")
    parser.add_argument('--model', type=str, default='models/resnet18_quantized.pth',
                        help='Path to the quantized model file.')
    parser.add_argument('--class_map', type=str, default='models/class_to_idx.json',
                        help='Path to the class map JSON file.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')

    args = parser.parse_args()

    predictor = ImagePredictor(model_path=args.model, class_map_path=args.class_map)
    predicted_class, confidence = predictor.predict(args.image)

    if predicted_class:
        print(f"Prediction: '{predicted_class}' with {confidence:.2%} confidence.")