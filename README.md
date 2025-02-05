# Property Value Prediction for Cook County

This project aims to predict property values in Cook County to ensure fair and equitable property tax assessments. By leveraging machine learning techniques and historical property data, the goal is to provide reliable estimates that aid in urban planning and taxation strategies.

## Project Overview

Cook County faces challenges with inconsistent property valuations, which can lead to imbalances in taxation and planning. Our solution involves using a Random Forest regression model to predict property values accurately by addressing missing data, feature engineering, and multi-collinearity.

---

## Key Objectives

1. **Develop a robust predictive model** for property assessments.
2. **Improve data quality** by addressing missing values and redundant columns.
3. **Provide key insights** to support decision-making for Cook County's urban planning and tax assessments.

---

## Data Sources

- **Historic Property Data**: Contains features like property characteristics, geographic information, and past sale prices.
- **Prediction Data**: New property records for which predictions are required.

---

## Methodology

### 1. **Data Cleaning and Preprocessing**
   - Handled missing values by imputing medians (numerical) and modes (categorical).
   - Removed columns with low correlation to the target variable (`sale_price`) and addressed multi-collinearity.

### 2. **Feature Engineering**
   - Encoded categorical variables into numerical form.
   - Scaled and normalized features to improve model performance.

### 3. **Model Training**
   - Utilized a Random Forest model for its robustness in handling non-linear relationships and large datasets.
   - Split the data: 80% for training, 20% for validation.
   - Achieved a **Mean Squared Error (MSE)** of 0.12302 on validation data.

### 4. **Prediction and Evaluation**
   - Generated predictions using the trained model.
   - Applied reverse transformations to provide interpretable sale price estimates.
   - Summary statistics:
     - **Mean Sale Price**: $350,000
     - **Standard Deviation**: $75,000
     - **Price Range**: $200,000 to $1,500,000

---

## Model Implementation

The Random Forest model was implemented in R. Key steps included:

- Handling missing values and feature scaling.
- Encoding categorical variables using label encoding.
- Splitting the data for training and validation.
- Predicting and scaling back the property values for final output.

### Code Overview

The R code consists of:
- Data loading and visualization.
- Missing data imputation and redundancy handling.
- Feature scaling, encoding, and Random Forest model training.
- Generating predictions for new property data.

---

## Results

- A CSV file (`predict_data_with_predictions.csv`) was generated, containing predicted property values alongside engineered features.
- The model showed strong performance, as evidenced by low error rates and high interpretability of results.

---

## Visualizations

- **Histogram of Sale Prices**: Displays the distribution of historic property values.
- **Boxplot**: Identifies outliers in property sale prices.
- **Feature Importance Chart**: Highlights key features affecting property values.

---

## Technologies Used

- **Programming Language**: R
- **Machine Learning Model**: Random Forest Regression
- **Libraries**: `randomForest`, `Metrics`, `caret`, `readr`

---

## Future Improvements

- Incorporate additional features, such as real-time economic indicators.
- Explore alternative models like Gradient Boosting or Neural Networks for improved performance.
- Implement hyperparameter tuning for further optimization.

---

## Contact

- **Author**: Nikhita Shankar  
- **Email**: [nikhita4@illinois.edu](mailto:nikhita4@illinois.edu)  
- **LinkedIn**: [Nikhita Shankar](https://linkedin.com/in/nikhita-shankar-)  

