(README is just a overview of project - check website post, technical report, or actual code for a more holistic understanding)
# Diamond Price Prediction

## Research Question
What physical characteristics of a diamond hold the most predictive value in its price, and can those factors be used to predict the value of diamonds in the consumer market?

## Dataset
- Source: [Kaggle Diamonds Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds)
- File used: `data/diamonds.csv`
- Rows: 53,940
- Target: `price`
- Features: `carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`
- Missing values: none

## EDA Findings
### Price Distribution
Most diamonds are in the lower price range. The distribution is right-skewed.

![Price Distribution](images/diamond%20price%20distribution.png)

### Numeric Features vs Price
`carat` has a strong positive relationship with price.
`x`, `y`, and `z` also show strong positive relationships.
`depth` and `table` show weaker relationships.

![Price vs Numeric Features](images/diamond%20Price%20vs%20Numeric%20Features.png)

### Correlation Matrix
Price is strongly correlated with `carat`, `x`, `y`, and `z`.

![Correlation Matrix](images/corr%20matrix%20for%20diamonds.png)

### Categorical Box Plots (Left to Right: Clarity, Color, Cut)
<table>
  <tr>
    <td><img src="images/box%20plot%20for%20clarity.png" alt="Clarity plot" width="100%"></td>
    <td><img src="images/box%20plot%20for%20color.png" alt="Color plot" width="100%"></td>
    <td><img src="images/box%20plot%20for%20cut.png" alt="Cut plot" width="100%"></td>
  </tr>
  <tr>
    <td align="center">Clarity</td>
    <td align="center">Color</td>
    <td align="center">Cut</td>
  </tr>
</table>

These plots show class balance for the main categorical features.

## Modeling
### Preprocessing
- Train and test split: 80/20
- Numeric features: `StandardScaler`
- Categorical features: `OneHotEncoder(handle_unknown='ignore')` then `StandardScaler(with_mean=False)`

`with_mean=False` keeps the one-hot matrix sparse and memory efficient.

### Base Models and Tuning
The following base models were tuned with `GridSearchCV`:
- ElasticNet
- LinearSVR
- KNN
- XGBoost

Setup:
- `cv=5`
- scoring: `neg_root_mean_squared_error`
- all models used the same preprocessing pipeline

For XGBoost, learning rate and number of trees were refined with early stopping on a validation split from the training data.

### Stacking Model
Final ensemble:
- Base models: tuned ElasticNet, LinearSVR, KNN, XGBoost
- Meta model: `RidgeCV`
- Stacking setting: `cv=5`

Stacking uses out-of-fold predictions from base models to train the meta model. The test set is only used at final evaluation.

### Permutation Importance
Permutation importance was used for interpretation.
This is a good choice for stacking because there is no single built-in feature importance attribute.

Method:
- shuffle one feature at a time
- measure the increase in RMSE
- larger increase means higher importance

## Model Performance
Best model: **Stacking**
- RMSE: **531.743517**
- R^2: **0.982213**

Model comparison:

![Model Results](images/diamond%20dataframe%20with%20results.png)

## Feature Importance
Top drivers from permutation importance:
1. `carat` (3339.956909)
2. `y` (1468.471681)
3. `clarity` (1101.310135)
4. `color` (765.001176)

Numeric importance output:

![Feature Importance Table](images/Diamond%20dataframe%20with%20Feature%20Importances.png)

Bar chart:

![Feature Importance Bar Chart](images/diamond%20Bar%20Chart%20of%20Importances.png)

## Conclusion
Stacking gave the best predictive performance on this dataset.
Size features are the strongest drivers of price, especially `carat` and `y`.
Quality features also matter, mainly `clarity` and `color`.
This aligns with the research question and supports using these features for consumer market price prediction.
