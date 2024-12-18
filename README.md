Heart Disease Detection
This project aims to predict the likelihood of heart disease in patients based on various health factors such as age, cholesterol levels, and blood pressure. It uses machine learning techniques to analyze the dataset and provide insights into the likelihood of heart disease, which can help in early detection and prevention.

Table of Contents
Installation
Dataset
Preprocessing
Modeling
Evaluation
Usage
Contributing
License
Installation
To run the project locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/tejus01/heart-disease-detection.git
Navigate into the project folder:

bash
Copy code
cd heart-disease-detection
Install the necessary dependencies. It’s recommended to use a virtual environment:

bash
Copy code
pip install -r requirements.txt
Dataset
The dataset used in this project is the Heart Disease UCI dataset, which contains information on patients' health attributes and the presence of heart disease. It includes the following features:

Age
Sex
Chest pain type
Blood pressure
Cholesterol
Fasting blood sugar
Resting electrocardiographic results
Maximum heart rate achieved
Exercise induced angina
ST depression induced by exercise
Slope of the peak exercise ST segment
Number of major vessels colored by fluoroscopy
Thalassemia
You can find the dataset here.

Preprocessing
Before training the model, the dataset undergoes the following preprocessing steps:

Handling missing values
Encoding categorical variables (e.g., sex, chest pain type)
Scaling numerical features for better model performance
Modeling
This project uses machine learning algorithms to predict heart disease. The models include:

Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
The models are evaluated based on accuracy, precision, recall, and F1 score.

Evaluation
After training the models, the performance is evaluated using metrics such as:

Accuracy
Confusion Matrix
ROC Curve
The best-performing model is selected for further testing and usage.

Usage
To predict heart disease for a new patient, you can run the following Python script:

bash
Copy code
python predict.py --age 60 --sex 1 --cp 3 --trestbps 130 --chol 250 --fbs 0 --restecg 0 --thalach 150 --exang 1 --oldpeak 1.5 --slope 2 --ca 0 --thal 2
This will return the likelihood of the patient having heart disease based on the model.

Contributing
Contributions are welcome! If you want to contribute to the project, please fork the repository, create a new branch, make your changes, and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
