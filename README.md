# Efficient Data Stream Anomaly Detection

## Project Overview

This project aims to detect anomalies in credit card transactions, leveraging advanced machine learning algorithms to identify potential fraud. By analyzing a simulated stream of transaction data, the system will learn to recognize unusual patterns that deviate from typical spending behavior. The primary focus is on enhancing fraud detection accuracy while minimizing false positives, ensuring secure and efficient financial transactions.

## Project Objectives

- Create a Python script that implements **Isolation Forest** and **Local Outlier Factor (LOF)** for anomaly detection in financial transactions.
- Evaluate and compare the performance of both algorithms in identifying fraudulent activities.
- Produce comprehensive statistics and visualizations on the detected anomalies to gain deeper insights into the patterns of fraudulent transactions.

## Dataset

The project utilizes the **Credit Card Fraud Detection** dataset sourced from Kaggle. This dataset comprises a variety of features that represent individual transactions, along with labels that indicate whether each transaction is fraudulent or legitimate. It serves as a critical foundation for training and evaluating the anomaly detection algorithms employed in this project.

### Data Preview

The dataset follows a structured format with the following columns:

```plaintext
Time, V1, V2, ..., V28, Amount, Class
0, -1.35980713, 1.19185711, ..., -0.05395037, 149.62, 0
1, -1.35835305, 2.29212882, ..., 0.25542586, 2.69, 0
2, -0.96662024, 0.36188434, ..., 0.19811233, 378.66, 0
...
```


## Getting Started

### Requirements
Before you begin, ensure that you have the following installed on your machine:

- **Python 3.x**: A programming language required to run the scripts.
- **pip**: The Python package manager for installing dependencies.

### Project Structure
The project directory is organized as follows:

```bash
Anomaly_detection_cobblestone/
│
├── anomaly_detection.py        # Main script for anomaly detection.
├── requirements.txt            # List of required Python packages.
├── .gitignore                  # Specifies files and directories to ignore in the repository.
├── README.md                   # Documentation for the project.
└── creditcard.csv              # Dataset containing credit card transactions.
```

### 2. Clone the Repository

To clone the repository, run the following command in your terminal:

```bash
git clone https://github.com/Srinu1976/Efficient-Data-Stream-Anomaly-Detection
```

### 3.Install Requirements
Navigate to the project directory and install the required packages using the following command:
```bash
cd Efficient-Data-Stream-Anomaly-Detection
```
### 4.Install Dependencies
Install the necessary Python packages using:
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

Credit Card Fraud Detection Dataset: Kaggle Dataset
	•	Isolation Forest: Documentation for Isolation Forest in scikit-learn
	•	Local Outlier Factor: Documentation for LOF in scikit-learn
	•	Matplotlib: Official Documentation for the Python Plotting Library
	•	Pandas: Data Analysis Library for Python
	•	NumPy: Fundamental Package for Numerical Computing in Python
	•	Online Anomaly Detection for Data Streams: StreamAD GitHub Repository
	•	Streaming Anomaly Detection Framework in Python: pysad GitHub Repository
	•	Research Papers: Awesome Anomaly Detection GitHub Repository

- **Credit Card Fraud Detection Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Isolation Forest**: [Documentation for Isolation Forest in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- **Local Outlier Factor**: [Documentation for LOF in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- **Matplotlib**: [Official documentation for the Python plotting library](https://matplotlib.org/stable/index.html)
- **Pandas**: [Data analysis library for Python](https://pandas.pydata.org/docs/index.html)
- **NumPy**: [Fundamental package for numerical computing in Python](https://numpy.org/doc/)
- **Online Anomaly Detection for Data Streams**: [StreamAD GitHub Repository](https://github.com/Fengrui-Liu/StreamAD)
- **Streaming Anomaly Detection Framework in Python**: [pysad GitHub Repository](https://github.com/selimfirat/pysad)
- **Research Papers**: [GitHub Repository](https://github.com/hoya012/awesome-anomaly-detection)
