# Human Voice Classification and Clustering

This project involves analyzing a dataset of extracted human voice features to perform two main tasks:

1.  **Classification:** Predict the gender (male/female) of a voice sample.
2.  **Clustering:** Group similar voice samples together based on their features.

The goal is to build, evaluate, and compare various machine learning models and deploy a simple application using Streamlit for predictions.

## Dataset

The dataset (`data/vocal_gender_features_new.csv`) contains 43 extracted audio features (like spectral centroid, pitch, MFCCs) and one target label column:

- `label`: 0 (Female), 1 (Male)

## Technologies Used

- **Python:** Core programming language.
- **Pandas, NumPy:** Data manipulation and numerical computing.
- **Scikit-learn:** Machine learning models (SVM, Random Forest, K-Means, DBSCAN) and preprocessing tools.
- **Matplotlib, Seaborn:** Data visualization.
- **Streamlit:** Web application framework for deployment.

## Getting Started

### Prerequisites

- Python 3.7 or later
- pip (Python package installer)

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/B4Mprofessor/Voice_Classification_Project
    cd Voice_Classification_Project
    ```

2.  **(Optional but Recommended) Create a Virtual Environment:**

    ```bash
    python -m venv venv
    # Activate the virtual environment:
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Analysis

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Navigate to and open `notebooks/Voice_Analysis.ipynb`.
3.  Run the cells in the notebook to perform data loading, cleaning, exploratory data analysis, train classification models (Logistic Regression, Random Forest, SVM), and perform clustering analysis (K-Means, DBSCAN).

### Running the Streamlit App

1.  Ensure you are in the project's root directory (`Voice_Classification_Project`).
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  The app should automatically open in your default web browser. If not, navigate to the URL provided in the terminal (usually `http://localhost:8501`).

## Key Findings

- **Classification:** The Support Vector Machine (SVM) model achieved the highest performance (Accuracy ~99.95%).
- **Clustering:** K-Means clustering was performed (optimal K=10 based on Silhouette Score). DBSCAN was attempted but did not find meaningful clusters with the tested parameters.
- **Application:** A Streamlit web application was successfully built to load the SVM model and make predictions based on manual input of the 43 features.

## Files

- `notebooks/Voice_Analysis.ipynb`: Contains the complete data analysis, model training, and clustering code.
- `app.py`: The Streamlit application script.
- `svm_model.pkl`: The saved, trained SVM model used by the Streamlit app.
- `scaler.pkl`: The saved feature scaler used by the Streamlit app.
