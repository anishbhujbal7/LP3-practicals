# 🚖 Uber Ride Fare Prediction

A machine learning project to predict Uber ride fares using Linear Regression and Random Forest Regression models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project aims to predict Uber ride fares based on various features including:
- Pickup and drop-off coordinates
- Distance traveled
- Time-related information (hour, day, month)
- Number of passengers

Two regression models are implemented and compared:
1. **Linear Regression** - A baseline statistical model
2. **Random Forest Regression** - An ensemble learning method

---

## 📊 Dataset

**Source:** [Kaggle – Uber Fares Dataset](https://www.kaggle.com/)

### Key Features:

| Column | Description |
|--------|-------------|
| `fare_amount` | Target variable (price of the ride in USD) |
| `pickup_datetime` | Timestamp when the ride started |
| `pickup_longitude` | Longitude coordinate of pickup location |
| `pickup_latitude` | Latitude coordinate of pickup location |
| `dropoff_longitude` | Longitude coordinate of drop-off location |
| `dropoff_latitude` | Latitude coordinate of drop-off location |
| `passenger_count` | Number of passengers in the ride |

---

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning models and evaluation
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

---

## 💾 Installation

### Prerequisites

Make sure you have Python 3.x installed on your system.

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/uber-fare-prediction.git
cd uber-fare-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place it in the `data/` directory.

---

## 📁 Project Structure

```
uber-fare-prediction/
│
├── data/
│   └── uber_fares.csv          # Dataset file
│
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook with EDA
│
├── src/
│   ├── preprocessing.py        # Data cleaning and feature engineering
│   ├── models.py               # Model implementation
│   └── evaluation.py           # Model evaluation functions
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── main.py                     # Main execution script
```

---

## 🔬 Methodology

### 1. Data Loading & Cleaning
- Loaded dataset using pandas
- Removed invalid entries:
  - Negative or zero fare amounts
  - Missing coordinate values
  - Null values in critical columns

### 2. Feature Engineering

**Distance Calculation (Haversine Formula):**

The great-circle distance between pickup and drop-off points:

```
d = 2r × arcsin(√(sin²(Δlat/2) + cos(lat₁)cos(lat₂)sin²(Δlon/2)))
```

Where:
- r = Earth's radius (6371 km)
- Δlat = difference in latitude
- Δlon = difference in longitude

**Time-Based Features:**
- `hour`: Hour of the day (0-23)
- `day_of_week`: Day of the week (0-6)
- `month`: Month of the year (1-12)
- `is_weekend`: Binary flag for weekend rides

### 3. Outlier Detection & Removal
- Removed fares below the 1st percentile
- Removed fares above the 99th percentile
- Ensures model isn't skewed by extreme values

### 4. Correlation Analysis
- Generated correlation heatmap
- Identified key features affecting fare amount

### 5. Model Training
- Split data: 80% training, 20% testing
- Trained Linear Regression model
- Trained Random Forest Regression model
- Applied cross-validation for robustness

### 6. Model Evaluation
- **R² Score**: Measures goodness of fit
- **RMSE**: Root Mean Squared Error for prediction accuracy

---

## 📈 Results

| Model | R² Score | RMSE |
|-------|----------|------|
| **Linear Regression** | 0.667 | 5.65 |
| **Random Forest Regression** | 0.720 | 5.18 |

### Key Findings:

✅ **Random Forest outperformed Linear Regression** by:
- Higher R² score (72% vs 66.7% variance explained)
- Lower RMSE (better prediction accuracy)
- Better handling of non-linear relationships
- Capturing complex feature interactions

📊 **Important Features:**
1. Distance (Haversine)
2. Hour of day
3. Day of week
4. Passenger count

---

## 🚀 Usage

### Running the Complete Pipeline

```bash
python main.py
```

### Using Individual Modules

**Preprocessing:**
```python
from src.preprocessing import clean_data, engineer_features

df_clean = clean_data('data/uber_fares.csv')
df_features = engineer_features(df_clean)
```

**Model Training:**
```python
from src.models import train_linear_regression, train_random_forest

lr_model = train_linear_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)
```

**Prediction:**
```python
# Predict fare for a new ride
prediction = rf_model.predict(new_ride_features)
print(f"Predicted Fare: ${prediction[0]:.2f}")
```

---

## 🔮 Future Improvements

- [ ] Implement XGBoost and Gradient Boosting models
- [ ] Add traffic and weather data as features
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Deploy model as a REST API using Flask/FastAPI
- [ ] Create a web interface for fare prediction
- [ ] Implement feature importance visualization
- [ ] Add surge pricing detection
- [ ] Incorporate more sophisticated outlier detection methods

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Anish Bhujbal**
- GitHub: [@yourusername](https://github.com/anishbhujbal7)


---

## 🙏 Acknowledgments

- Kaggle for providing the Uber fares dataset
- scikit-learn community for excellent documentation
- All contributors and supporters of this project

---

## 📞 Contact

For questions or feedback, please open an issue or reach out via email.

---

**⭐ If you found this project helpful, please consider giving it a star!**