# Virus Classification using CNN

This repository contains a deep learning project for classifying virus images using Convolutional Neural Networks (CNNs).

## Project Description
The goal of this project is to classify microscopic virus images into multiple classes using transfer learning models such as EfficientNet and ResNet.

## Dataset
- Virus Image Dataset (Kaggle)
- The dataset is already pre-split into:
  - Training
  - Validation
  - Test
- Number of classes: 22

## Data Preparation
- Image resizing to 224×224
- Normalization (rescale 1./255)
- Data augmentation:
  - Rotation
  - Zoom
  - Horizontal flip

## Contributors
- **Malak** – Dataset preparation, preprocessing, augmentation, and data loaders

## Notes
All large datasets are loaded directly from Kaggle and are not uploaded
