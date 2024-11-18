Malaria Cell Image Classification using Deep Learning
Project Overview
This project implements a Convolutional Neural Network (CNN) for analyzing malaria cell images. The model uses VGG16 for feature extraction followed by a custom classifier to determine whether a cell is parasitized or uninfected.
Dataset

Source: NIH Malaria Dataset
Content: Cell images divided into two categories

Parasitized: Malaria infected cells
Uninfected: Normal cells


Dataset Link

Project Structure
├── cell_images/
│   ├── Parasitized/
│   └── Uninfected/
├── training/
├── validation/
├── testing/
├── build_dataset.py
├── train_model.py
├── saved_models/
└── README.md

Implementation Details

Data Preprocessing

Split ratio: 80% training, 10% validation, 10% testing
Image rescaling (1/255.0)
Target size: 150x150 pixels
RGB color mode


Model Architecture

Feature Extractor: VGG16 (pre-trained on ImageNet)
Custom Classifier:
pythonCopySequential([
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])



Training

Optimizer: Adam (learning_rate=0.0001)
Loss: Binary Crossentropy
Batch Size: 32
Epochs: 50
Early Stopping: patience=10



Performance

Target Accuracy: 93.7%
Achieved Test Accuracy: 94.67%
Additional Metrics:

Precision: 0.94
Recall: 0.93
F1-score: 0.93



Requirements
Copytensorflow
numpy
matplotlib
seaborn
scikit-learn
Installation & Usage

Clone the repository

bashCopygit clone [repository-url]

Install dependencies

bashCopypip install -r requirements.txt

Run the data split script

bashCopypython build_dataset.py

Run the training script

bashCopypython train_model.py
Results Visualization

Training/Validation accuracy and loss curves
Confusion Matrix
Prediction probability distribution

Model Saving
The model is saved in two formats:

Best model during training: 'best_model.keras'
Final model: 'final_malaria_model.keras'
