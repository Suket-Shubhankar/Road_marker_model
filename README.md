# Road Marker Model

The "Road Marker Model" project leverages deep learning to detect and mark roads in given images. The model is designed using PyTorch and is made available for inference using the [Cog](https://github.com/replicate/cog) framework.

## Requirements

- **Python Version**: 3.11.5
- **GPU**: This model requires a GPU for inference.
- **CUDA Version**: 12.1

### Python Packages:

- numpy==1.25.2
- torch==2.1.0
- torchvision==0.16.0
- pandas==2.1.1
- scikit-learn==1.3.1

## Model Architecture

The model, `RoadDetection`, is a convolutional neural network designed specifically for road detection. The architecture comprises of convolutional layers, ReLU activations, max-pooling layers, and transposed convolutional layers, culminating in a sigmoid activation for binary segmentation.

## Using the Model with Cog

The model is set up to be used with Cog, enabling easy and streamlined deployment. To run predictions using this model:

1. Set up Cog following the instructions in the [official documentation](https://github.com/replicate/cog).
2. Use the `predict.py` script which defines the `Predictor` class for model inference. This script includes methods for setting up the model, preprocessing the input images, running predictions, and postprocessing the model's output.

## Quick Start

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
