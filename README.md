# Heart Disease Prediction WebService

This repository contains the final project for DATA 1200, which integrates machine learning with web services to predict heart disease. The project trains three models—**Random Forest**, **SVM** (both supervised), and **K-means** (unsupervised clustering)—on a heart disease dataset, then deploys these models via a RESTful API using Flask. In addition, a professional, interactive web page is provided for users to enter patient data and receive predictions.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [References and Resources](#references-and-resources)
- [Authors and Acknowledgments](#authors-and-acknowledgments)
- [License](#license)

---

## Overview

This project predicts the likelihood of heart disease in patients using three different machine learning algorithms:
- **Random Forest Classifier**
- **SVM Classifier**
- **K-means Clustering Model**

A Jupyter Notebook (`train_models.ipynb`) performs exploratory data analysis (EDA), pre-processes the data (including one-hot encoding of categorical variables), trains and evaluates the models, and saves the trained models as pickle files. 

The Flask API (`app.py`) loads these models and provides endpoints for predictions. An interactive and professionally styled front-end web page (`templates/index.html`) allows end users to enter patient data and receive a diagnosis—with clear output indicating whether heart disease is detected (binary outcome 1 or 0) along with a text message.

---

## Project Structure

DATA-1200-02-Heart-Disease-Prediction-WebService/
├── app.py                   # Flask web app
├── train_models.ipynb       # Notebook for training models
├── feature_names.pkl        # .pkl model file
├──kmeans_model.pkl          # .pkl model file
├──random_forest_model.pkl   # .pkl model file
├──svm_model.pkl             # .pkl model file
├── templates/               # HTML templates for the web app
├──heart.csv                 # Dataset for the project
└── README.md                # Project documentation

---

## Features

- **Machine Learning Models:**
  - Random Forest and SVM classifiers for supervised prediction.
  - K-means clustering (with a mapping to a binary diagnosis) for unsupervised analysis.
- **RESTful API:**  
  Endpoints:
  - `/randomforest/evaluate` – returns the binary prediction and diagnosis message.
  - `/svm/evaluate` – returns the binary prediction and diagnosis message.
  - `/kmeans/evaluate` – maps cluster output to a binary prediction with a diagnosis message.
- **Interactive Web Interface:**  
  A responsive HTML page built with Bootstrap guides the user with placeholder recommendations for input ranges and displays clear output (both text and binary prediction).
- **Reproducible Environment:**  
  All code is versioned, and the project’s folder structure and files satisfy the professor’s requirements.

---

# Installation and Setup

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- Git and VS Code (optional but recommended)
- Required Python packages:
  - Flask
  - NumPy
  - Pandas
  - scikit-learn
  - matplotlib

To install the required packages, run:

```bash
pip install flask numpy pandas scikit-learn matplotlib
```
# Setting Up the Project Locally
## 1. Clone the Repository

```bash
git clone https://github.com/Shriram002/DATA-1200-02-Heart-Disease-Prediction-WebService.git

cd DATA-1200-02-Heart-Disease-Prediction-WebService
```
## 2. Open in VS Code
   Launch the folder in Visual Studio Code or another IDE.

## 3. Train the Models
   Open and run all cells in train_models.ipynb to:
   - Load the dataset
   - Perform data preprocessing & EDA
   - Train Random Forest, SVM, and K-means models
   - Save them as .pkl files
## 4. Run the Flask Web App

```bash
python app.py
```

Then open http://127.0.0.1:5000/ in your browser to interact with the app.


---

# Usage

Follow the steps below to use the web interface:

1. **Start the Flask App**

   Run the command below from your terminal:

   ```bash
   python app.py 
    ```

2. **Open the Web Interface**
   Navigate to http://127.0.0.1:5000/ in your web browser.
   
3. **Enter Input Values**
   - Fill out the patient’s health details.
   - Each field provides a suggested value range.
   - Examples:
     - Age: 30–80
     - RestingBP: 90–200
     - Cholesterol: 100–600
     - Binary fields: 0 = No, 1 = Yes
       
4. **Choose a Model**
   - Select from:
       - Random Forest
       - Support Vector Machine (SVM)
       - K-means Clustering
         
5. **View Results**
   - The prediction result will be displayed clearly.
   - Output includes:
        - Binary value: 1 = Heart Disease, 0 = No Heart Disease
        - Text message (e.g., “Heart Disease Detected”)

# References and Resources

Below is a list of references and resources used in this project:

##  Libraries and Tools

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Bootstrap](https://getbootstrap.com/)

##  Dataset Source

- **Heart Disease Dataset**  
  Source: [Kaggle / UCI Heart Disease](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

##  Code Inspiration

- Official sklearn examples for model training and evaluation
- Tutorials on deploying ML models with Flask and pickle

# Authors and Acknowledgments

##  Team Members

- **Shriram**  
- **Anshul**  
- **Nikita**  
- **Tanya**

Each member contributed equally in areas of:
- Data preprocessing
- Model training
- Web API development
- Frontend design
- Testing and deployment

##  Special Thanks

We would like to thank **Professor Shanti Couvrette** for the DATA-1200-02 course for her continued guidance and support throughout the semester.

# License

This project is licensed under name of Shriram Yadav.

You are free to:
- Use, copy, modify, merge, publish, and distribute the software

Under the following conditions:
- Include the original copyright
- Include the license text in any copies


