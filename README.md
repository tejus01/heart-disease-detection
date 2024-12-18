# Heart Disease Detection using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

Early detection of heart disease is crucial for effective treatment and prevention. This project leverages machine learning techniques to predict the likelihood of heart disease in patients based on various health factors. By analyzing a patient's medical data, our models provide valuable insights that can aid in early diagnosis and intervention.

## Getting Started

### Prerequisites

Before running the project, ensure you have Python 3.7 or higher installed. It's highly recommended to create a virtual environment to manage project dependencies.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/tejus01/heart-disease-detection.git](https://github.com/tejus01/heart-disease-detection.git)
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd heart-disease-detection
    ```

3.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv  # For macOS/Linux
    python -m venv venv    # For Windows
    ```

4.  **Activate the virtual environment:**

    ```bash
    source venv/bin/activate   # For macOS/Linux
    venv\Scripts\activate     # For Windows
    ```

5.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

This project utilizes the Heart Disease UCI dataset, a widely used dataset in machine learning for heart disease prediction. It contains 76 attributes, but all published studies refer to using a subset of 14 of them. These features include:

*   Age
*   Sex
*   Chest pain type (4 values)
*   Resting blood pressure
*   Serum cholestoral in mg/dl
*   Fasting blood sugar > 120 mg/dl
*   Resting electrocardiographic results (values 0,1,2)
*   Maximum heart rate achieved
*   Exercise induced angina
*   ST depression induced by exercise relative to rest
*   The slope of the peak exercise ST segment
*   Number of major vessels (0-3) colored by flourosopy
*   Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

You can find the dataset [here](https://archive.ics.uci.edu/ml/datasets/heart+Disease).

## Data Preprocessing

The dataset undergoes several preprocessing steps to prepare it for model training:

*   **Handling Missing Values:** Missing values (if any) are handled using appropriate imputation techniques (e.g., mean/median imputation or more advanced methods).
*   **Encoding Categorical Variables:** Categorical features like 'Sex' and 'Chest pain type' are converted into numerical representations using one-hot encoding or label encoding.
*   **Feature Scaling:** Numerical features are scaled using StandardScaler or MinMaxScaler to ensure that all features contribute equally to the model training process, preventing features with larger values from dominating.

## Machine Learning Models

The following machine learning models are used for heart disease prediction:

*   **Logistic Regression:** A linear model for binary classification.
*   **Random Forest Classifier:** An ensemble learning method that combines multiple decision trees.
*   **Support Vector Machine (SVM):** A powerful algorithm that finds the optimal hyperplane to separate data points.
*   **K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies data points based on their proximity to neighbors.

Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. A confusion matrix and ROC curve analysis are also performed to select the best-performing model.

## Usage

To predict the likelihood of heart disease for a new patient, use the `predict.py` script:

```bash
python predict.py --age 60 --sex 1 --cp 3 --trestbps 130 --chol 250 --fbs 0 --restecg 0 --thalach 150 --exang 1 --oldpeak 1.5 --slope 2 --ca 0 --thal 2
