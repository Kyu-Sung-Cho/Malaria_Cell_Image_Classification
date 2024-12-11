# **Malaria Cell Image Classification Project**

# About
A motivated data scientist with a proven track record of leveraging data to solve complex problems and driving impactful business decisions.

Extensive cross-industry experience in ESG consulting, manufacturing, healthcare, and academia, showcasing expertise in data analysis, predictive modeling, and building data-driven applications. Proficient in Python, SQL, R, Tableau, and Power BI, combined with a strong foundation in machine learning, NLP, and statistical modeling. Adept at creating ETL pipelines, crafting personalized recommendation systems, and developing dashboards for actionable insights, while excelling in stakeholder communication and cross-functional collaboration.

Keen passion for advancing data science applications in healthcare and sustainability. Successfully fine-tuned LLM models for radiology annotation, optimized forecasting models for revenue growth, and designed a green index to enhance ESG performance. Always seeking innovative solutions to bridge technical and business goals while fostering impactful collaboration.

# Education
+ M.S., Applied Data Science | The University of Chicago 
(Sep 2023 – Present)

+ M.S., Business and Technology Management | Korea Advanced Institute of Science and Technology 
(Aug 2020 – Aug 2022)

+ B.S., Security and Risk Analysis (Cybersecurity) | Pennsylvania State University 
(Aug 2014 – May 2018)

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
- **Source**: NIH Malaria Dataset ([link](https://ceb.nlm.nih.gov/repositories/malaria-datasets/))  
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

## **Repository Contents**

- **Scripts**:
  - `malaria_classification.py`: Full pipeline for preprocessing, feature extraction, model training, and evaluation.  
- **Models**:
  - `best_malaria_model.keras`: Best model saved during training.  
  - `final_malaria_model.keras`: Final trained model.  

---

## **Conclusion**
This project highlights the transformative potential of AI in healthcare by automating malaria diagnosis through image classification. The success of this model emphasizes the importance of transfer learning and feature extraction in building robust, scalable solutions for real-world problems.
