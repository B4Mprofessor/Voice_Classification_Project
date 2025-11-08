# Human Voice Classification and Clustering

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.1%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.18%2B-red.svg)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-green.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A machine learning project to classify human voice gender (Male/Female) and cluster similar voices using extracted audio features.**

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Methodology](#-methodology)
- [Models](#-models)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 🧠 Overview

This project demonstrates a complete machine learning pipeline for audio signal analysis. The primary goal is to build and evaluate models capable of automatically determining the gender (Male/Female) of a human voice sample based on 43 pre-extracted audio features. Additionally, the project explores unsupervised learning by clustering voice samples based on their feature similarity. The best-performing classification model is deployed via a user-friendly web interface built with Streamlit.

## 📊 Dataset

The dataset (`vocal_gender_features_new.csv`) contains **16,147 samples** (after initial cleaning) with **44 columns**:

- **43 Features:** Numerical values representing various characteristics extracted from audio files.
- **1 Label:** Binary target variable (`0` for Female, `1` for Male).

**Key Feature Categories:**

- Spectral Features (e.g., centroid, bandwidth, contrast, flatness, rolloff)
- Pitch Statistics (mean, min, max, std)
- Other Features (zero_crossing_rate, rms_energy, spectral_skew, spectral_kurtosis)
- Mel-Frequency Cepstral Coefficients (MFCCs) - Mean and Standard Deviation (1st to 13th coefficients)

> **Note:** The last row of the original CSV file contained column names as data, which was handled during the loading process.

## ⚙️ Features

- **Data Preprocessing:** Handles missing values, outlier removal (using IQR), and feature standardization.
- **Exploratory Data Analysis (EDA):** Statistical summaries, correlation analysis, and feature distribution visualization.
- **Model Development:**
  - **Classification:** Train and compare Logistic Regression, Random Forest, and Support Vector Machine (SVM) models.
  - **Clustering:** Apply K-Means and DBSCAN clustering algorithms.
- **Model Evaluation:** Compare models using Accuracy, Precision, Recall, F1-Score (classification) and Silhouette Score (clustering).
- **Model Deployment:** Deploy the best model (SVM) using a Streamlit web application.

## 🧪 Methodology

1.  **Data Loading & Cleaning:**
    - Load the dataset, correctly handling the header row issue.
    - Convert data types.
    - Identify and remove rows with missing labels/features.
    - Detect and remove outliers using the Interquartile Range (IQR) method.
2.  **Exploratory Data Analysis (EDA):**
    - Analyze label distribution (found to be slightly imbalanced: ~64% Male, ~36% Female).
    - Visualize feature distributions and correlations.
3.  **Feature Scaling:**
    - Apply `StandardScaler` to normalize features for algorithms sensitive to feature magnitude (e.g., SVM, K-Means).
4.  **Train-Test Split:**
    - Split the data into training and testing sets, maintaining the label distribution (stratified split).
5.  **Model Training:**
    - Train Logistic Regression, Random Forest, and Support Vector Machine (SVM) classifiers.
    - Apply K-Means clustering (finding optimal K using Silhouette Score) and attempt DBSCAN.
6.  **Model Evaluation:**
    - Evaluate classification models on the test set using Accuracy, Precision, Recall, and F1-Score.
    - Evaluate clustering results using Silhouette Score and Calinski-Harabasz Score.
7.  **Model Selection & Deployment:**
    - Select the best performing model (SVM with F1-Score ~0.9995).
    - Save the trained model and the fitted scaler using `joblib`.
    - Build a Streamlit web application (`app.py`) to load the model and make predictions.

## 🤖 Models

- **Logistic Regression:** Baseline linear model.
- **Random Forest:** Ensemble method using decision trees.
- **Support Vector Machine (SVM):** Powerful classifier, achieved the highest performance.
- **K-Means Clustering:** Partitional clustering algorithm.
- **DBSCAN Clustering:** Density-based clustering algorithm (attempted, but did not find clear clusters with tested parameters).

## 📈 Results

- **Classification:**
  - **Best Model:** Support Vector Machine (SVM)
  - **Performance:**
    - **Accuracy:** ~99.95%
    - **Precision:** ~0.9995
    - **Recall:** ~0.9995
    - **F1-Score:** ~0.9995
- **Clustering:**
  - **K-Means:** Optimal K found using Silhouette Score (e.g., K=10).
  - **DBSCAN:** Did not find meaningful clusters with initial parameter sweeps.
- **Deployment:** A functional Streamlit app is available to load the SVM model and make predictions based on manual feature input.

## 🛠️ Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/B4Mprofessor/Voice_Classification_Project.git
    cd human-voice-classification-clustering
    ```

2.  **(Optional but Recommended) Create a Virtual Environment:**

    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _(Ensure `requirements.txt` contains the necessary packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib`)_

## 🚀 Usage

### For Analysis (`Voice_Analysis.ipynb`)

1.  Ensure the `data/vocal_gender_features_new.csv` file is present in the `data` directory.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Navigate to and open `notebooks/Voice_Analysis.ipynb`.
4.  Run the cells sequentially to reproduce the entire analysis pipeline.

### For Deployment (`app.py`)

1.  Ensure `svm_model.pkl` and `scaler.pkl` are present in the main project directory (created after running the analysis notebook).
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).
4.  Enter the 43 voice feature values manually into the input fields.
5.  Click the "Classify Voice" button to get the predicted gender.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or feature additions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Thanks to the source of the `vocal_gender_features_new.csv` dataset.
- Inspired by the principles of machine learning and audio signal processing.

---

<div align="center">

Made with ❤️ using Python, Scikit-learn, and Streamlit.

</div>
