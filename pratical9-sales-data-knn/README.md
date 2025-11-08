# 🧮 K-Means Clustering on Sales Data

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📘 Objective

The objective of this practical is to implement the K-Means clustering algorithm on the `sales_data_sample.csv` dataset to group sales records into distinct clusters based on sales and pricing patterns. We also determine the optimal number of clusters using the Elbow Method.

---

## 📂 Dataset

**Dataset link:** [Sales Data Sample – Kaggle](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)

### Key Features Used

After exploring the dataset, the following numerical features were selected for clustering:

- `QUANTITYORDERED` – Number of items ordered
- `PRICEEACH` – Price per item
- `SALES` – Total sales amount
- `MSRP` – Manufacturer's Suggested Retail Price
- `QTR_ID` – Quarter of the year
- `MONTH_ID` – Month of the year
- `YEAR_ID` – Year of transaction

---

## ⚙️ Steps Involved

### 1️⃣ Importing and Reading Data

```python
import pandas as pd

df = pd.read_csv('/content/sales_data_sample.csv', encoding='latin1')
```

### 2️⃣ Selecting Relevant Columns

```python
df = df[['QUANTITYORDERED','PRICEEACH','SALES','MSRP','QTR_ID','MONTH_ID','YEAR_ID']]
```

### 3️⃣ Data Preprocessing

- Handled missing values if any.
- Scaled data using StandardScaler to normalize feature ranges:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
```

### 4️⃣ Determining Optimal Number of Clusters (Elbow Method)

The Elbow Method was used to identify the optimal `k` value by plotting the inertia (sum of squared distances from points to their assigned cluster centers).

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal k')
plt.show()
```

✅ From the graph, the "elbow point" occurs around **k = 4**.

### 5️⃣ Applying K-Means Clustering

```python
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

### 6️⃣ Cluster Analysis

Cluster-wise averages were calculated to interpret patterns in each group:

```python
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)
```

**Sample Output:**

| Cluster | QUANTITYORDERED | PRICEEACH | SALES | MSRP | QTR_ID | MONTH_ID | YEAR_ID |
|---------|----------------|-----------|-------|------|--------|----------|---------|
| 0 | 32.30 | 73.19 | 2373.41 | 78.81 | 1.43 | 3.19 | 2004.18 |
| 1 | 35.09 | 97.91 | 4430.29 | 123.93 | 3.64 | 9.87 | 2003.52 |
| 2 | 33.58 | 63.10 | 2099.50 | 66.87 | 3.61 | 9.80 | 2003.54 |
| 3 | 40.92 | 98.46 | 5448.43 | 131.89 | 1.47 | 3.37 | 2004.23 |

---

## 📊 Interpretation

- **Cluster 0:** Moderate-priced sales, mostly in early 2004 (Q1).
- **Cluster 1:** High-value sales in late 2003 (Q3–Q4).
- **Cluster 2:** Low-value sales, also in late 2003.
- **Cluster 3:** Premium high-sales orders, early 2004.

This shows seasonal and value-based grouping in the sales data.

---

## 📈 Insights

- Two time-based patterns emerge — **2003** (Clusters 1 & 2) and **2004** (Clusters 0 & 3).
- **Cluster 3** contains the most profitable transactions.
- **Cluster 2** likely represents low-price or bulk discount sales.
- The clustering can help in market segmentation, sales forecasting, and targeted marketing.

---

## 🧠 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `sklearn.preprocessing`
- `sklearn.cluster`

---

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/anishbhujbal7/kmeans-sales-clustering.git
   cd kmeans-sales-clustering
   ```

2. Place the dataset (`sales_data_sample.csv`) in the project directory.

3. Run the notebook or Python script:
   ```bash
   jupyter notebook kmeans_clustering.ipynb
   ```

---

## 📁 Project Structure

```
kmeans-sales-clustering/
│
├── data/
│   └── sales_data_sample.csv
│
├── kmeans_clustering.ipynb
├── README.md
└── requirements.txt
```

---

## ✅ Conclusion

K-Means clustering successfully identified **4 distinct sales behavior patterns** in the dataset. The Elbow Method helped determine the optimal number of clusters, revealing meaningful patterns in sales performance across time and price ranges.

---

## 👨‍💻 Author

**Anish Bhujbal**  
Machine Learning Practical — K-Means Clustering on Sales Data  
Course: LP3 — Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-anishbhujbal7-black?logo=github)](https://github.com/anishbhujbal7)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle – Sample Sales Data](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)
- Course: LP3 — Machine Learning