# **Malaria Cell Image Classification Project**

## **Overview**
This project demonstrates the application of transfer learning with the VGG16 model to classify malaria-infected cells from healthy ones using a dataset of microscopic images. By designing and training a convolutional neural network (CNN), the study emphasizes how artificial intelligence can assist in diagnosing critical diseases like malaria efficiently and accurately.

---

## **Business Problem**
Malaria remains a significant global health issue, particularly in low-resource settings where diagnostic tools are limited. The primary objective of this project is to develop an AI-powered model that can:
- Accurately classify cell images as “Parasitized” or “Uninfected.”
- Reduce dependency on manual microscopy diagnostics.
- Enhance early detection and treatment planning.

---

## **Dataset and Features**

### **Dataset Overview**
- **Source**: NIH Malaria Dataset
- **Total Images**: ~27,558 cell images  
- **Classes**:  
  - Parasitized: Images of malaria-infected cells.  
  - Uninfected: Images of healthy cells.  

### **Data Preparation**
1. **Data Splitting**:
   - Training Set: 80%  
   - Validation Set: 10%  
   - Test Set: 10%  
2. **Image Preprocessing**:
   - Rescaled pixel values to [0, 1] range using `ImageDataGenerator`.  
   - Standardized image dimensions to 150x150 pixels.  

---

## **Model Architecture**

### **Transfer Learning with VGG16**
- **Pre-trained Weights**: ImageNet.  
- **Feature Extraction**:
  - Removed the top fully connected layers.  
  - Extracted features with the convolutional base for all dataset splits.  

### **Custom Classifier**
- **Input**: Flattened feature vectors from the VGG16 model.  
- **Architecture**:
  - Fully Connected Layer 1: 512 units, ReLU activation, Dropout (50%).  
  - Fully Connected Layer 2: 256 units, ReLU activation, Dropout (30%).  
  - Output Layer: 1 unit, Sigmoid activation for binary classification.  

---

## **Training Process**

1. **Hyperparameters**:
   - Optimizer: Adam (learning rate = 0.0001)  
   - Loss Function: Binary Crossentropy  
   - Batch Size: 32  
   - Epochs: 100 (with early stopping)  

2. **Callbacks**:
   - Early Stopping: Patience = 20 epochs, monitored validation accuracy.  
   - Model Checkpoint: Saved the best model during training.  

3. **Data Augmentation**:
   - Augmented training images for better generalization.  

---

## **Key Findings**

### **Performance Metrics**
- **Test Accuracy**: 94.78%, exceeding the target accuracy of 93.7%.  
- **Loss**:
  - Training Loss: Converged steadily with minimal overfitting.  
  - Validation Loss: Stabilized effectively over epochs.  

### **Confusion Matrix Insights**
| **Metric**         | **Value**   |
|---------------------|-------------|
| True Positives      | 2,651       |
| True Negatives      | 2,573       |
| False Positives     | 153         |
| False Negatives     | 135         |

### **Classification Report**
- Precision and Recall were high for both “Parasitized” and “Uninfected” classes, reflecting robust model performance.

### **Visualizations**
1. **Accuracy and Loss Curves**:
   - **Accuracy**: Training and validation accuracy curves demonstrated good convergence.  
   - **Loss**: Training loss decreased steadily, with validation loss remaining low.  

2. **Confusion Matrix**:
   - Visualized with Seaborn’s heatmap for easy interpretation.  

3. **Prediction Probabilities**:
   - Histogram plotted for the predicted probabilities of the “Parasitized” class.  

---

## **Results Summary**

| **Metric**         | **Value**   |
|---------------------|-------------|
| Test Accuracy       | 94.78%      |
| Target Accuracy     | 93.7%       |
| True Positives      | 2,651       |
| False Positives     | 153         |
| False Negatives     | 135         |

**Key Insight**: The model demonstrated excellent performance and exceeded expectations, achieving a high accuracy with minimal overfitting.

---

## **Limitations & Future Work**

### **Limitations**
- Lack of explainability for model predictions.  
- Performance might vary with unseen datasets due to limited domain-specific data.  

### **Future Directions**
1. Integrate Grad-CAM to highlight regions in images contributing most to model decisions.  
2. Experiment with more advanced architectures (e.g., ResNet, EfficientNet).  
3. Expand the dataset to include other cell-based diseases for multi-class classification tasks.  

---

## **Conclusion**
This project highlights the transformative potential of AI in healthcare by automating malaria diagnosis through image classification. The success of this model emphasizes the importance of transfer learning and feature extraction in building robust, scalable solutions for real-world problems.
