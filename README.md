# Heart Disease Prediction Using Machine Learning

Multi-class classification of heart disease severity (0–4) using classical ML models trained on combined clinical datasets.

## Dataset

Three publicly available heart disease datasets combined:
- Cleveland Processed
- Hungarian Reprocessed
- UCI Heart Disease

**Total samples:** 3,290 (after cleaning and upsampling)  
**Features:** 13 clinical attributes (age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope, number of major vessels, thalassemia)  
**Target:** 5 classes (0 = no disease, 1–4 = increasing severity)

## Preprocessing

- Combined three datasets with consistent column naming
- Removed rows with missing values (encoded as `?`)
- Upsampled minority classes (2, 3, 4) to match the largest class using random oversampling
- Applied MinMaxScaler for models sensitive to feature scale

## Models and Results

| Model | Test Accuracy |
|---|---|
| Random Forest | **95.29%** |
| XGBoost | 94.68% |
| Decision Tree | 93.92% |
| Gradient Boosting | 92.40% |
| Logistic Regression | 50.46% |
| AdaBoost | 49.85% |

**Best model:** Random Forest Classifier with 95.29% test accuracy on 5-class classification.

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Scikit-plot

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib scikit-plot
jupyter notebook Heart_Disease_Pre_2__1_.ipynb
```

Update the dataset file paths in the notebook to point to your local copies of the datasets.
