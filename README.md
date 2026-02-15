
# Chest X-Ray Pneumonia Classification using ResNet-18

## Project Overview

This project builds a deep learning model to classify chest X-ray images into two categories:

* **NORMAL**
* **PNEUMONIA**

We use **Transfer Learning** with a pre-trained ResNet-18 model to leverage learned visual features and fine-tune it for medical image classification.

The model is trained using PyTorch and evaluated using Accuracy and F1 Score.

---

## Objectives

Use transfer learning for medical image classification
Train a binary classifier using chest X-ray dataset
Evaluate model performance
Predict class for new unseen images

---

## Model Architecture

* Base Model: **ResNet-18 (Pretrained)**
* Frozen feature extractor layers
* Custom fully connected layer for binary classification
* Sigmoid activation for probability output
* Loss Function: BCEWithLogitsLoss
* Optimizer: Adam

---

## Dataset Structure

```
data/
 ├── chestxrays/
      ├── train/
      │     ├── NORMAL/
      │     ├── PNEUMONIA/
      ├── test/
            ├── NORMAL/
            ├── PNEUMONIA/
```

### Create virtual environment 

```bash
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install torch torchvision torchmetrics numpy pillow
```

---

## Training the Model

Run the training script:

```bash
python train.py
```

The model will:

* Load dataset
* Apply normalization
* Train for 3 epochs
* Print training loss and accuracy

---

## Saving Model

After training, save the model weights:

```python
torch.save(model.state_dict(), "chest_xray_resnet18.pth")
```

---

## Model Evaluation

The model is evaluated on the test dataset using:

* Accuracy
* F1 Score

Example output:

```
Test accuracy: 0.91
Test F1-score: 0.90
```

---

## Predict on New Image

Run prediction script:

```bash
python predict.py
```

Or call function:

```python
predict_image("test_xray.jpg")
```

Output:

```
Prediction: PNEUMONIA
Probability of Pneumonia: 0.8734
```

---

## Image Preprocessing

Images are normalized using ImageNet statistics:

```
Mean = [0.485, 0.456, 0.406]
Std = [0.229, 0.224, 0.225]

```

Images are resized to **224 × 224** before inference.


## Metrics Used

* Binary Accuracy
* F1 Score

These metrics help evaluate classification performance, especially for imbalanced datasets.


## Tool Used

* Python
* PyTorch
* Torchvision
* Torchmetrics
* NumPy
* PIL

