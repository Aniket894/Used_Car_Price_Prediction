
### Project Documentation: Used Car Price Prediction  
(ML Regression Project)

---

**Table of Contents**  
1. Introduction  
2. Dataset Description  
3. Project Objectives  
4. Project Structure  
5. Data Ingestion  
6. Data Transformation  
7. Model Training  
8. Training Pipeline  
9. Prediction Pipeline  
10. Flask  
11. Logging  
12. Exception Handling  
13. Utils  
14. Conclusion  

---

### 1. Introduction  
The Car Price Prediction project aims to predict the price of cars based on various features such as car name, year, distance driven, fuel type, and other characteristics. This document provides a comprehensive overview of the project, including its structure, processes, and supporting scripts.

---

### 2. Dataset Description  
**Dataset Name:** Car Price Dataset  

**Description:** The dataset contains 8015 entries and 9 columns, providing various features that can help in predicting car prices:  
- **Car Name:** The name of the car.  
- **Year:** The year the car was manufactured.  
- **Distance:** The total distance driven by the car (in kilometers).  
- **Owner:** The number of previous owners the car has had.  
- **Fuel:** The type of fuel the car uses (e.g., Petrol, Diesel).  
- **Location:** The geographical location where the car is being sold.  
- **Drive:** The drive type of the car (e.g., Front-Wheel Drive, Rear-Wheel Drive).  
- **Type:** The type of car (e.g., Sedan, SUV).  
- **Price:** The final selling price of the car.

---

### 3. Project Objectives  
- **Data Ingestion:** Load and explore the dataset.  
- **Data Transformation:** Clean, preprocess, and transform the dataset for analysis.  
- **Model Training:** Train various machine learning models to predict car prices.  
- **Pipeline Creation:** Develop a pipeline for data ingestion, transformation, and model training.  
- **Supporting Scripts:** Provide scripts for setup, logging, exception handling, and utilities.

---

### 4. Project Structure  
```
├── artifacts/
│   ├── (best)model.pkl
│   ├── linearRegression.pkl
│   ├── Lasso.pkl
│   ├── Ridge.pkl
│   ├── ElasticNet.pkl
│   ├── DecisionTreeRegressor.pkl
│   ├── RandomForestRegressor.pkl
│   ├── GradientBoostingRegressor.pkl
│   ├── AdaBoostRegressor.pkl
│   ├── XGBoostRegressor.pkl
│   ├── KNeighborsRegressor.pkl
│   ├── raw.csv
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── data/
│   │   └── car_24_Combined.csv
│   └── Car_Price_Prediction(2).ipynb
│
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_training.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── static/
│   ├── car.jpeg
│   └── style.css
│
├── app.py
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py
```

---

### 5. Data Ingestion  
The data ingestion module loads the car dataset, splits it into training and testing sets, and saves them as CSV files. The raw data is stored in the `artifacts/` folder for future reference.

---

### 6. Data Transformation  
The data transformation module handles preprocessing, including encoding categorical variables (e.g., Car Name, Fuel, Location, Drive, Type) and scaling numerical variables (e.g., Year, Distance, Owner, Price). The transformed data is stored in the `artifacts/` folder.

---

### 7. Model Training  
The model training module trains multiple machine learning regression models such as:  
- **Linear Regression**  
- **Lasso Regression**  
- **Ridge Regression**  
- **ElasticNet**  
- **Decision Tree Regressor**  
- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  
- **AdaBoost Regressor**  
- **XGBoost Regressor**  
- **KNeighbors Regressor**

The best-performing model is saved as `best_model.pkl` in the `artifacts/` folder.


![model_adjusted_r2_comparison](https://github.com/user-attachments/assets/b154142c-50fb-49b8-8599-c77d1851ec72)

---

### 8. Training Pipeline  
The training pipeline module integrates data ingestion, data transformation, and model training, ensuring that all components are executed in the correct sequence, from loading the data to saving the best model.

---

### 9. Prediction Pipeline  
The prediction pipeline uses `best_model.pkl` and `preprocessor.pkl` to predict car prices on new data. It handles preprocessing and model inference seamlessly.

---

### 10. Flask (app.py)  
The Flask app (`app.py`) provides a web interface for users to input car details and receive price predictions. The form inputs are collected in `index.html`, and the results are displayed in `results.html`.


![Screenshot 09-03-2024 09 18 11](https://github.com/user-attachments/assets/bbab5469-86f5-44cf-8349-b165cb6a2386)



![Screenshot 09-03-2024 09 21 29](https://github.com/user-attachments/assets/5ae94f28-e4cb-49e4-91e7-89843e5d8678)


---

### 11. Logging  
The `logger.py` file captures logs during project execution, including data ingestion, transformation, model training, and errors encountered. The logs are stored in a designated folder for debugging and monitoring.

---

### 12. Exception Handling  
The `exception.py` file contains exception handling code, ensuring that errors in the pipeline are caught and logged, which helps maintain the robustness of the project.

---

### 13. Utils  
The `utils.py` file contains utility functions for repetitive tasks like directory creation, file management, and data loading.

---

### 14. Conclusion  
This documentation outlines the complete workflow of the Car Price Prediction project, covering the ingestion, transformation, and modeling processes. The project's modular structure allows for easy maintenance, scalability, and adaptability for future use cases.

---
