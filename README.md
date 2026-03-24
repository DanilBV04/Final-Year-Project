# Image Classification with Deep Convolutional Neural Network

A deep learning project built with TensorFlow and Keras implementing a 5-layer Convolutional Neural Network (CNN) for image classification, complete with a data pipeline, training visualisation, and a Flask API endpoint.

## Overview

This project implements an end-to-end image classification pipeline including custom data loading, preprocessing, augmentation, model training, and result visualisation. The architecture was designed to handle real-world image datasets with class imbalance and variable image quality.

## Architecture

The model consists of 5 convolutional blocks with progressively increasing filter sizes, followed by fully connected layers:

```
Input (400x400x3)
→ Conv2D(16, 3x3, ReLU) → MaxPooling(2x2)
→ Conv2D(32, 3x3, ReLU) → MaxPooling(2x2)
→ Conv2D(64, 3x3, ReLU) → MaxPooling(2x2)
→ Conv2D(128, 3x3, ReLU) → MaxPooling(2x2)
→ Conv2D(256, 3x3, ReLU) → MaxPooling(2x2)
→ Flatten → Dropout(0.5)
→ Dense(60, ReLU, L2) → Dropout(0.5)
→ Dense(60, Sigmoid, L2)
→ Dense(1, Sigmoid)
```

## Features

- **Custom data pipeline** — modular image loading and preprocessing with error handling for corrupt files
- **Data augmentation** — rotation, width/height shifting, shear, zoom, and horizontal flipping to improve generalisation
- **Class weight balancing** — automatic computation of class weights to handle imbalanced datasets
- **Adaptive learning rate** — ReduceLROnPlateau callback reduces learning rate on plateaus
- **Training visualisation** — accuracy and loss curves plotted for both training and validation sets
- **Misclassification analysis** — visualises misclassified samples for model debugging
- **Flask API** — REST endpoint for model inference

## Project Structure

```
├── main.py              # CNN model definition, training loop, and visualisation
├── fileProcessor.py     # Image loading and preprocessing utilities
├── fileDownloader.py    # Dataset directory management and image counting
```

## Technologies

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- NumPy
- Pillow (PIL)
- Matplotlib
- Flask

## Getting Started

### Prerequisites

```bash
pip install tensorflow scikit-learn numpy pillow matplotlib flask
```

### Running the Model

1. Place your image dataset in the directory specified in `fileDownloader.py`
2. Run the training pipeline:

```bash
python main.py
```

3. Training will run for 25 epochs with learning rate reduction on validation loss plateau
4. Accuracy/loss curves and misclassified samples will be displayed on completion

## Results

The model outputs:
- Training and validation accuracy curves per epoch
- Training and validation loss curves per epoch
- Visual grid of misclassified samples with predicted vs true labels
- Final test accuracy score

## Key Design Decisions

- **400x400 input resolution** — preserves fine-grained features in image data
- **L2 regularisation** on Dense layers to reduce overfitting
- **Dropout(0.5)** at two points in the network for additional regularisation
- **Class weight computation** ensures minority classes are not ignored during training
- **Modular structure** separates data handling from model logic for maintainability

## Future Work

- Extend to multi-class classification with softmax output
- Integrate transfer learning (ResNet, EfficientNet) for improved accuracy
- Expand Flask API to full REST interface with model serving
- Apply architecture to medical image analysis for early detection of genetic conditions in paediatric datasets
