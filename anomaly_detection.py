# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pylab import rcParams

# Set plot parameters for visualization
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42  # Seed for reproducibility
LABELS = ["Normal", "Fraud"]  # Labels for class distribution

# Function to load the dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

# Function to show basic dataset info
def dataset_overview(df):
    print("\n--- Dataset Overview ---")
    print(df.info())
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    return df['Class'].value_counts()

# Function to sample the data for faster processing
def sample_data(df, fraction=0.1, seed=RANDOM_SEED):
    df_sample = df.sample(frac=fraction, random_state=seed)
    print(f"\nSample shape: {df_sample.shape}, Original shape: {df.shape}")
    return df_sample

# Function to perform anomaly detection
def anomaly_detection(X, Y, outlier_fraction):
    classifiers = {
        "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(X),
                                            contamination=outlier_fraction, random_state=RANDOM_SEED),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
    }
    
    for clf_name, clf in classifiers.items():
        if clf_name == "Local Outlier Factor":
            y_pred = clf.fit_predict(X)
            y_pred = np.where(y_pred == -1, 1, 0)  # Adjust LOF predictions
        else:
            clf.fit(X)
            y_pred = clf.predict(X)
            y_pred = np.where(y_pred == -1, 1, 0)  # Adjust Isolation Forest predictions
        
        # Calculate number of misclassifications
        n_errors = (y_pred != Y).sum()
        print(f"\n--- {clf_name} Results ---")
        print(f"Number of errors: {n_errors}")
        print(f"Accuracy Score: {accuracy_score(Y, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(Y, y_pred))

# Function to plot class distribution
def plot_class_distribution(count_classes):
    count_classes.plot(kind='bar', rot=0)
    plt.title("Transaction Class Distribution")
    plt.xticks(range(2), LABELS)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load dataset
    file_path = 'creditcard.csv'  # Adjust path if necessary
    df = load_dataset(file_path)
    
    if df is not None:
        # Get overview of dataset
        count_classes = dataset_overview(df)
        
        # Split dataset into Fraud and Normal transactions
        fraud = df[df['Class'] == 1]
        normal = df[df['Class'] == 0]
        print(f"\nFraud transactions: {fraud.shape}, Normal transactions: {normal.shape}")
        
        # Sample the dataset for faster processing
        df_sample = sample_data(df)
        
        # Determine fraud/valid transaction counts in the sample
        Fraud = df_sample[df_sample['Class'] == 1]
        Valid = df_sample[df_sample['Class'] == 0]
        outlier_fraction = len(Fraud) / float(len(Valid))
        print(f"\nOutlier fraction: {outlier_fraction}")
        
        # Create independent (X) and dependent (Y) features
        columns = [col for col in df_sample.columns if col != 'Class']
        X = df_sample[columns]
        Y = df_sample['Class']
        print(f"\nFeatures shape: {X.shape}, Target shape: {Y.shape}")
        
        # Perform anomaly detection
        anomaly_detection(X, Y, outlier_fraction)
        
        # Plot class distribution
        plot_class_distribution(count_classes)