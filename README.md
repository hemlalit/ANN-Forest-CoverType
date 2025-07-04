# 🌲 Forest Cover Type Classification using ANN

This project utilizes an Artificial Neural Network (ANN) to classify forest cover types based on various cartographic and ecological attributes. The model is trained on the well-known Forest CoverType dataset from `sklearn.datasets`, making it an excellent real-world example of multi-class classification.

## 📂 Project Structure

- `ARTIFICIAL NEURAL NETWORK – Forest CoverType.ipynb`: Main Jupyter notebook containing EDA, model development, and evaluation.
- `fetch_covtype`: Dataset used for training and testing the model from `sklearn.datasets`.
- `README.md`: Project overview and documentation (you’re reading it!).

## 🧠 Objective

The goal is to predict forest cover types (7 classes) using 54 numerical and binary features such as elevation, aspect, slope, soil type, and wilderness area.

## ⚙️ Workflow

1. **Data Preprocessing**
   - Handled missing values (if any).
   - Standardized continuous features.
   - One-hot encoded categorical features.

2. **Exploratory Data Analysis**
   - Scaled data using StandardScaler.
   - Analyzed correlations and feature importance.

3. **Model Architecture**
   - Implemented using Keras with TensorFlow backend.
   - Input Layer: 54 features
   - Hidden Layers: Multiple Dense layers with ReLU activation
   - Output Layer: Softmax activation for 7 classes

4. **Evaluation**
   - Metrics: Accuracy, Confusion Matrix, Classification Report
   - Achieved 87% accuracy

## 🔍 Key Insights

- Elevation was the most informative feature.
- ANN performed significantly better than naive models.
- Class imbalance needed to be accounted for in loss function and metrics.

## 📦 Libraries & Tools

- Python (NumPy, Pandas)
- TensorFlow / Keras
- Matplotlib, Seaborn
- Scikit-learn
