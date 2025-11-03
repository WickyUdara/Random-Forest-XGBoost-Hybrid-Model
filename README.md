# Sri Lanka Climate: Max Temperature Prediction

This project analyzes climate data from Sri Lanka to predict the maximum daily temperature (`temperature_2m_max`). It explores and compares four different machine learning approaches to determine the most effective model.

The analysis is contained within the `submission (2).ipynb` Jupyter Notebook.

## Dataset

The dataset used is the **Sri Lanka Climate Data** sourced from [Kaggle](https://www.kaggle.com/).

## Methodology

The project follows these main steps:

### Data Loading & Preprocessing

- Loads the `.csv` file and sorts it by date to preserve temporal order.
- Performs feature engineering by extracting `year`, `month`, `day`, `dayofweek`, and `dayofyear` from the date.
- Handles missing values for `temperature_2m_min` and `precipitation_sum` by filling them with the median.

### Exploratory Data Analysis (EDA)

- A correlation matrix heatmap is generated to visualize relationships between the numeric features.

### Model Training & Evaluation

- The data is split into training (80%) and testing (20%) sets using a time-based split (`shuffle=False`).
- An evaluation function is defined to report RMSE, MAE, and R² scores.

#### Four Different Models Trained and Evaluated:

1. **Random Forest Regressor**: A baseline model using `RandomForestRegressor` from scikit-learn.

2. **XGBoost Regressor**: An implementation using `XGBRegressor`.

3. **Hybrid (RF + ANN)**: This approach uses the predictions from the trained Random Forest model as an additional feature for an Artificial Neural Network (built with Keras/TensorFlow).

4. **Hybrid (XGB + ANN)**: Similar to the RF hybrid, this model uses the XGBoost predictions as an input feature for the ANN.

### Comparison & Conclusion

All models are compared based on their test set performance (RMSE, MAE, and R²).

The notebook includes visualizations comparing the predicted values from each model against the actual values.

## Key Findings

Based on the analysis in the notebook:

- **XGBoost Model**: Best performance with an R² of approximately **0.707**
- **Hybrid (XGB + ANN) Model**: Second-best performance with an R² of **0.662**
- **Random Forest Model**: Third place with an R² of **0.637**
- **Hybrid (RF + ANN) Model**: Weakest performance with an R² of **0.590**

**Conclusion**: The standalone XGBoost model was the most effective for this dataset. While adding its predictions to an ANN improved upon the RF model, it did not surpass the performance of XGBoost itself.

## How to Use

### Dependencies

Ensure you have the following libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `joblib`
- `scikit-learn`
- `xgboost`
- `tensorflow`

### Run

1. Download the `sri_lanka_climate.csv` file from the Kaggle link above and place it in the same directory.
2. Run the cells in the `submission (2).ipynb` notebook sequentially.

### Saved Models

The notebook saves the trained models to the following files:

- `rf_model.joblib`
- `xgb_model.joblib`
- `rf_ann_hybrid.h5`
- `xgb_ann_hybrid.h5`

---
