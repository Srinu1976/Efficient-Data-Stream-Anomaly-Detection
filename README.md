# Efficient Data Stream Anomaly Detection

## Overview
This project focuses on detecting anomalies in financial transactions, particularly credit card transactions, using advanced machine learning techniques. By simulating a continuous data stream, the goal is to identify unusual patterns that may indicate fraudulent activity.

## Objectives
- Develop a Python script that utilizes **Isolation Forest** and **Local Outlier Factor (LOF)** to detect anomalies.
- Analyze the effectiveness of both algorithms in identifying fraudulent transactions.
- Generate detailed statistics on the detected anomalies for better insights.

## Dataset
The project employs the **Credit Card Fraud Detection** dataset from Kaggle. It contains various features representing transactions, with labels indicating whether they are fraudulent or legitimate.

### Data Preview
Sample data format:
```plaintext
Time, V1, V2, ..., V28, Amount, Class
0, -1.35980713, 1.19185711, ..., -0.05395037, 149.62, 0
1, -1.35835305, 2.29212882, ..., 0.25542586, 2.69, 0
2, -0.96662024, 0.36188434, ..., 0.19811233, 378.66, 0
...
```

## Getting Started

### Requirements
Ensure that you have the following installed on your machine:

- Python 3.x
- pip (Python package manager)

### Project Structure
```bash
Anomaly_detection_cobblestone/
│
├── anomaly_detection.py        # The main script for anomaly detection.
├── requirements.txt            # A list of required Python packages.
├── .gitignore                  # A file that specifies which files and directories to ignore in the repository.
├── README.md                   # This file.
└── creditcard.csv              # The dataset for credit card transactions.
```
#### Setting Up the Project

### 1.Download the Dataset

1. Go to the Kaggle dataset page: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Sign in to your Kaggle account (or create one if you don't have an account).
3. Click on the "Download" button to download the dataset as a ZIP file.
4. Extract the ZIP file, and place the `creditcard.csv` file in the project directory.

### 2.Clone the Repository
To clone the repository, use the following command:
```bash
git clone https://github.com/Lohit2005/FinancialAnomalyTracker
```

### 3.Install Requirements
Navigate to the project directory and install the required packages using the following command:
```bash
cd FinancialAnomalyTracker
```
### 4.Install Dependencies
```bash
pip install -r requirements.txt
```

### Running the Project
You can run the anomaly detection script using the command line:
```bash
python anomaly_detection.py
```

### Results and Statistics
After running the script, you will see the output showing the number of anomalies detected by each algorithm. The following statistics will be printed:
```bash
Anomaly Detection Statistics:
Isolation Forest - Detected Anomalies: X
Local Outlier Factor - Detected Anomalies: Y
```

### References

- **Credit Card Fraud Detection Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Isolation Forest**: [Documentation for Isolation Forest in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- **Local Outlier Factor**: [Documentation for LOF in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- **Matplotlib**: [Official documentation for the Python plotting library](https://matplotlib.org/stable/index.html)
- **Pandas**: [Data analysis library for Python](https://pandas.pydata.org/docs/index.html)
- **NumPy**: [Fundamental package for numerical computing in Python](https://numpy.org/doc/)
- **Online Anomaly Detection for Data Streams**: [StreamAD GitHub Repository](https://github.com/Fengrui-Liu/StreamAD)
- **Streaming Anomaly Detection Framework in Python**: [pysad GitHub Repository](https://github.com/selimfirat/pysad)
- **Research Papers**: [GitHub Repository](https://github.com/hoya012/awesome-anomaly-detection)
