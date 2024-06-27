# Churn Modelling Project

This project aims to build and evaluate a deep learning model to predict customer churn using Keras and Scikit-Learn. The project includes data preprocessing, model training, evaluation, and visualization of the results.

Author: Kaíque Freire dos Santos<br>
Date: 2024/06/26

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Script Overview](#script-overview)
- [Results](#results)
- [Dependencies](#dependencies)

## Project Structure
```
├── .gitignore
├── LICENSE
├── README.md
├── dataset-variables/
│ ├── churn-modelling.pkl
│ ├── churn-modelling.zip
├── logs/
│ ├── info-preprocessing.log
│ ├── training.log
├── plots/
│ ├── pie-exited.png
│ ├── count-per-country.png
│ ├── count-per-gender.png
│ ├── count-per-product.png
│ ├── correlation-map-numerics.png
│ ├── plot-training-validation.png
│ ├── confusion-matrix.png
├── save_model/
│ ├── model.h5
│ ├── weights_model.weights.h5
├── .spyproject/
│ ├── config
├── kaggle.json
├── model.py
├── preprocessing.py
```
## Installation

1. Clone the repository:
```
   git clone https://github.com/kaiquefreire05/py-churn-modelling.git
```
2. Navigate to the project directory:
```
   cd py-churn-modelling
```
4. Create and activate a virtual environment:
```
  python -m venv env
  source env/bin/activate # On Windows use env\Scripts\activate
  ```
6. Install the required dependencies:
```
  pip install -r requirements.txt
  ```

## Usage

1. **Preprocess the data:**
  ```
  python preprocessing.py
  ```

This script will load, clean, visualize, and transform the data, saving it for model training.

2. **Train and evaluate the model:**
  ```
   python model.py
  ```
This script will load the preprocessed data, define and train the deep learning model, generate performance charts, evaluate the model, and save the trained model and its weights.

## Script Overview

### Model Training and Evaluation (model.py)

This module handles the training and evaluation of a deep learning model using Keras and Scikit-Learn.

#### Main Components:
- **LoggingCallback**: Custom callback to log metrics during model training.
- **create_model()**: Defines and returns a deep learning model with the specified architecture.
- **Main Execution**: 
- Loads training and testing data.
- Sets up the logging system.
- Defines and compiles the model.
- Trains the model with callbacks for early stopping, learning rate reduction, and logging.
- Generates accuracy and loss charts.
- Evaluates the model using accuracy metrics, classification report, and confusion matrix.
- Saves the trained model and its weights.

### Data Preprocessing (preprocessing.py)

This module performs data preprocessing for the deep learning project.

#### Main Components:
- **Data Loading**: Loads data from a zip file into a DataFrame.
- **Data Cleaning**: Removes unnecessary columns.
- **Data Visualization**: Generates charts to understand data distribution.
- **Data Transformation**: Converts categorical columns using OneHot encoding and normalizes numerical data.
- **Data Splitting**: Splits the data into training and testing sets.
- **Save Data**: Saves the processed data using pickle.

## Results

- **Training/Validation Accuracy and Loss**: Charts showing model performance during training.
- **Confusion Matrix**: Visual representation of model predictions vs. actual labels.
- **Accuracy Score**: Overall accuracy of the model.
- **Classification Report**: Detailed classification metrics.

## Dependencies

- pandas
- numpy
- scikit-learn
- keras
- matplotlib
- seaborn
- logging
- zipfile
- pickle

Ensure all dependencies are installed using the provided `requirements.txt`.

