# 🏦 Bank Customer Churn Prediction (Neural Network)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📘 Objective

The goal of this project is to build a neural network–based classifier that predicts whether a bank customer is likely to leave (churn) within the next 6 months.

---

## 📊 Dataset

**Source:** [Kaggle - Bank Customer Churn Modeling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

The dataset contains **10,000 customer records** with **14 features**, including:

- `CustomerId`, `Surname`
- `CreditScore`
- `Geography` (France, Germany, Spain)
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- **`Exited`** (Target variable — 1 if the customer left, 0 if retained)

---

## ⚙️ Steps Performed

### 1️⃣ Data Preprocessing

* **Dropped unnecessary columns:** `RowNumber`, `CustomerId`, `Surname`
* **Converted categorical features** into numeric form using one-hot encoding:
  ```python
  df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
  ```
  (e.g., "France" becomes 0/1 binary column)
* **Encoded Gender** (Male → 1, Female → 0)
* **Scaled numerical columns** using `StandardScaler` to normalize feature ranges:
  ```python
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```

### 2️⃣ Model Building

A Sequential Neural Network was built using TensorFlow/Keras:

```python
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

* **Input layer:** Takes all feature values
* **Hidden layers:** Use ReLU activation for non-linearity
* **Dropout:** Prevents overfitting (drops 20% of neurons randomly)
* **Output layer:** Sigmoid activation outputs probability of churn (0–1)

### 3️⃣ Compilation and Training

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

* **Optimizer:** Adam (efficient gradient-based optimization)
* **Loss Function:** Binary Cross-Entropy (for binary classification)
* **Metric:** Accuracy
* **Epochs:** 50 (iterations over training data)
* **Validation Split:** 20% of training data for validation

### 4️⃣ Model Evaluation

Training and validation performance:

| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | ~0.86 | ~0.85 |
| **Loss** | ~0.34 | ~0.33 |

✅ Both curves show smooth convergence — no overfitting observed.  
✅ The model generalizes well to unseen data.

### 5️⃣ Performance Visualization

**Model Accuracy and Loss Curves:**
* Accuracy increases steadily and stabilizes near 85%.
* Loss decreases smoothly, indicating effective learning.

---

## 📈 Results Summary

* The neural network achieved **~85% accuracy** on validation data.
* Both training and validation losses converged properly.
* Model effectively distinguishes customers likely to churn vs. stay.

---

## 💡 Conclusion

The ANN model successfully predicts customer churn with high accuracy and balanced performance. The approach can help banks identify at-risk customers and take proactive retention measures.

---

## 🧰 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Scikit-learn**

---

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/anishbhujbal7/bank-customer-churn-prediction.git
   cd bank-customer-churn-prediction
   ```

2. Place the dataset (`Churn_Modelling.csv`) in the project directory.

3. Run the notebook or Python script:
   ```bash
   jupyter notebook bank_churn_prediction.ipynb
   ```

---

## 📁 Project Structure

```
bank-customer-churn-prediction/
│
├── data/
│   └── Churn_Modelling.csv
│
├── bank_churn_prediction.ipynb
├── README.md
└── requirements.txt
```

---

## 👨‍💻 Author

**Anish Bhujbal**  
Machine Learning Practical – Neural Network (Bank Churn Classification)  
Course: LP3 – Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-anishbhujbal7-black?logo=github)](https://github.com/anishbhujbal7)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* Dataset: [Kaggle - Bank Customer Churn Modeling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
* Course: LP3 – Machine Learning