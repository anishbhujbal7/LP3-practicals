# 🩺 Diabetes Prediction using K-Nearest Neighbors (KNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📘 Objective

The goal of this experiment is to predict whether a person has diabetes based on medical diagnostic data using the K-Nearest Neighbors (KNN) classification algorithm. We also apply hyperparameter tuning to optimize the model and evaluate it using various metrics.

---

## 📊 Dataset Description

**Source:** [Kaggle – Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The dataset contains **768 patient records** with **8 input features** and **1 output label** (Outcome).

| Feature | Description |
|---------|-------------|
| **Pregnancies** | Number of times pregnant |
| **Glucose** | Plasma glucose concentration (2-hour test) |
| **BloodPressure** | Diastolic blood pressure (mm Hg) |
| **SkinThickness** | Triceps skinfold thickness (mm) |
| **Insulin** | 2-hour serum insulin (mu U/ml) |
| **BMI** | Body mass index (kg/m²) |
| **DiabetesPedigreeFunction** | Genetic likelihood of diabetes |
| **Age** | Age in years |
| **Outcome** | Target (1 = Diabetic, 0 = Non-Diabetic) |

---

## ⚙️ Methodology

### 1️⃣ Import and Explore the Dataset

```python
import pandas as pd
df = pd.read_csv("diabetes.csv")
df.info()
df.describe()
```

Checked for nulls, datatypes, and class balance.

### 2️⃣ Correlation Analysis

```python
import numpy as np
corr = df.select_dtypes(include=np.number).corr()
```

🧠 **Key insights:**
- **Glucose (0.466)** — strongest correlation with diabetes
- **BMI (0.292)**, **Age (0.238)**, and **Pregnancies (0.222)** are also significant predictors

### 3️⃣ Split Dataset into Training and Testing Sets

```python
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4️⃣ Data Scaling

KNN relies on distance measures, so feature scaling is critical.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 5️⃣ Model Training — Basic KNN

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

### 6️⃣ Evaluation Metrics

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
error_rate = 1 - accuracy
```

**Example output:**

```
Confusion Matrix:
[[89 11]
 [18 36]]

Accuracy: 0.81
Error Rate: 0.19
Precision: 0.77
Recall: 0.67
```

### 7️⃣ Visualization — Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()
```

### 8️⃣ Hyperparameter Tuning using GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)
```

**Example output:**

```
Best Parameters: {'metric': 'manhattan', 'n_neighbors': 9, 'weights': 'distance'}
Best CV Accuracy: 0.845
```

### 9️⃣ Evaluate Tuned Model

```python
best_knn = grid.best_estimator_
y_pred_best = best_knn.predict(X_test)

print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_best))
```

**Example:**

```
Before tuning:  Accuracy = 0.81  
After tuning:   Accuracy = 0.85 ✅
```

---

## 📈 Final Results

| Metric | Before Tuning | After Tuning |
|--------|---------------|--------------|
| **Accuracy** | 0.81 | 0.85 |
| **Precision** | 0.77 | 0.80 |
| **Recall** | 0.67 | 0.73 |
| **Error Rate** | 0.19 | 0.15 |

✅ Hyperparameter tuning improved all performance metrics.

---

## 💡 Conclusion

The K-Nearest Neighbors algorithm effectively predicts diabetes with an accuracy of **~85%** after optimization.

- **Glucose, BMI, and Age** are the most significant factors.
- Using **GridSearchCV** helped find the best k, distance metric, and weighting, improving model generalization and performance.

---

## 🧰 Tools & Libraries

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Seaborn, Matplotlib**

---

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/anishbhujbal7/diabetes-prediction-knn.git
   cd diabetes-prediction-knn
   ```

2. Place the dataset (`diabetes.csv`) in the project directory.

3. Run the notebook or Python script:
   ```bash
   jupyter notebook diabetes_knn.ipynb
   ```

---

## 📁 Project Structure

```
diabetes-prediction-knn/
│
├── data/
│   └── diabetes.csv
│
├── diabetes_knn.ipynb
├── README.md
└── requirements.txt
```

---

## 👨‍💻 Author

**Anish Bhujbal**  
Machine Learning Practical — KNN on Diabetes Dataset (with Hyperparameter Tuning)  
Course: LP3 — Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-anishbhujbal7-black?logo=github)](https://github.com/anishbhujbal7)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle – Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Course: LP3 — Machine Learning