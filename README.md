# Cancer Detection Using CNN & VGG19

## Overview

This project is a deep learning-based classification system that differentiates between benign and malignant cancer images using Convolutional Neural Networks (CNNs) and VGG19. The model is trained on a dataset of skin cancer images and utilizes transfer learning combined with an Inception block to improve accuracy.

## Technologies Used
- **Python**
- **TensorFlow & Keras**
- **VGG19 Transfer Learning**
- **Inception Module**
- **Matplotlib for Visualization**
- **Scikit-learn for Metrics**

## Project Structure
📂 SkinCancer-Detection  
 ┣ 📂 data  
 ┃ ┣ 📂 train (Training images)  
 ┃ ┗ 📂 test (Test images)  
 ┣ 📜 model.py (CNN & VGG19 model implementation)  
 ┣ 📜 train.py (Training script)  
 ┣ 📜 evaluate.py (Model evaluation & metrics)  
 ┣ 📜 README.md (Project overview)  
 ┗ 📜 requirements.txt (Dependencies)  


## How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/silakbpc/Skin-Cancer-Classification.git
   cd Skin-Cancer-Classification
2. **Install dependencies**
  pip install -r requirements.txt
3. **Train the model**
   python train.py
4. **Evaluate the model**
   python evaluate.py

## Results & Performance

The model achieves high accuracy in classifying benign and malignant cancer images.
Performance is evaluated using confusion matrix, sensitivity, and specificity metrics.

## Contributors
silakbpc
