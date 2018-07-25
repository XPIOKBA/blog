---
layout: post
title: Housing Prices in Ames
summary: Did you overpay for your house in Iowa? Find out!
---
_[Git](https://github.com/afriedman412/ames)_  
[The Ames Housing data set](http://www.amstat.org/publications/jse/v19n3/decock.pdf) was introduced in 2011 by Dean De Cock at Truman State University. It contains 2930 observations of 80 variables, divided more or less evenly among ordinal, nominal, continuous and discrete. With data from 2006-2010, De Cock intended to provide a contemporary alternative to the Boston Housing data set, which was compiled in the 70's.

In this notebook, I will clean and explore the data, then briefly optimize a handful of regression models for predicting 'Sale Price'.

<br>

# Cleaning the Data
---
Fixing typos found earlier

```python
df.loc[1699,'Garage Yr Blt'] = 2007
df.loc[1712,'Garage Type'] = np.NaN
df.loc[1712,'Garage Cars'] = 0
df.loc[1712,'Garage Area'] = 0
```
<br>
Dropping 'Id' and 'PID' columns, which will not be needed

```python
df.drop(['Id','PID'], axis=1, inplace=True)
```

<br>

### Converting categorical features to numerical features
A number of categorical features are clearly ordered, with qualitative values like 'Good', 'Fair', 'Poor', etc. These are converted into ordered, numerical equivalents using a dictionary of keys.


```python
# features to be converted
to_cat = ['Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Heating QC', 'Kitchen Qual','Functional',
          'Fireplace Qu', 'Garage Qual', 'Garage Cond', 'Pool QC', 'Central Air', 'Garage Finish']

# keys for conversions
cat_key = {'five':
           {'Gd':3, 'TA':2, 'Ex':4, 'Po': 0, 'Fa':1},
           'five_NA':
           {'Gd':4, 'TA':3, 'Ex':5, 'Po': 1, 'Fa':2, 'N':0, np.NaN:0},
           'function':
           {'Typ':7, 'Mod':4, 'Min2':5, 'Maj1':3, 'Min1':6, 'Sev':1, 'Sal':0, 'Maj2':2},
           'binary':
           {'N':0, 'Y':1},
           'finish':
           {'RFn':2, 'Unf':1, 'Fin':3, np.NaN:0}}

# assigning each column a key dict
num_key = {'Exter Qual':'five', 'Exter Cond':'five', 'Bsmt Qual':'five_NA',
           'Bsmt Cond':'five_NA', 'Heating QC':'five', 'Kitchen Qual':'five',
           'Functional':'function', 'Fireplace Qu':'five_NA', 'Garage Qual':'five_NA',
           'Garage Cond':'five_NA', 'Pool QC':'five_NA', 'Central Air':'binary',
           'Garage Finish':'finish'}

# converting
for c in to_cat:
    p = str(f'{c}_num')
    df[p] = df[c].map(lambda x: cat_key[num_key[c]][x])
```

<br>

### Converting years to ages
A number of features list years (of construction, renovation, etc). To better standardize the data, they are converted to ages. 


```python
# features to be converted
to_age = ['Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Yr Sold']

# new feature names
age_names = ['House Age', 'Remod/Add Age', 'Garage Age', 'Years Since Sale']

# dictionary of old names and new names
ager = dict(zip(to_age,age_names))

# convert by subtracting from current year
for n, v in ager.items():
    df[v] = df[n].map(lambda x: 2018-x)
```

<br>

### Removing converted features


```python
df.drop([c for c in to_age + to_cat], axis=1, inplace=True)
```

<br>

### Filling in missing values


```python
nulls = {'col':[], 'null':[]}
for c in df.columns:
    if df[c].isnull().sum() > 0:
        nulls['col'].append(c)
        nulls['null'].append(df[c].isnull().sum())
        
pd.DataFrame(nulls)
```

| **Feature** | **# of Null Values** |
| :---: | :---: |
| Lot Frontage | 330 |
| Alley | 1911 |
| Mas Vnr Type | 22 |
| Mas Vnr Area | 22 |
| Bsmt Exposure | 58 |
| BsmtFin Type 1 | 55 |
| BsmtFin SF 1 | 1 |
| BsmtFin Type 2 | 56 |
| BsmtFin SF 2 | 1 |
| Bsmt Unf SF | 1 |
| Total Bsmt SF | 1 |
| Bsmt Full Bath | 2 |
| Bsmt Half Bath | 2 |
| Garage Type | 114 |
| Fence | 1651 |
| Misc Feature | 1986 |
| Garage Age | 114 |

Given that we only have 2051 rows, dropping 'Alley', 'Fence' and 'Misc Feature'.

```python
df.drop(['Alley', 'Fence', 'Misc Feature'], axis=1, inplace=True)
```

<br>

Filling in missing 'Garage Age' values with same from 'House Age'.

```python
df['Garage Age'].fillna(df['House Age'], inplace=True)
```

<br>

Imputing missing values for remaining features with column means for numericals and 'Unknown' for categoricals.

```python
# recalculating null columns
nulls = {'col':[], 'null':[]}
for c in df.columns:
    if df[c].isnull().sum() > 0:
        nulls['col'].append(c)
        nulls['null'].append(df[c].isnull().sum())

# imputing missing values
for c in nulls['col']:
    if df[c].dtype == 'object':
        df[c].fillna('Unknown', inplace=True)
    if df[c].dtype == 'float64':
        df[c].fillna(df[c].mean(), inplace=True)
```

<br>

### Dummy columns
Convert unordered categorical features to numerical by making dummy columns

```python
df_dummies = pd.get_dummies(df)
```

<br>

# Exploring the Data
---
Some visualizations and observations about the data. (Sometimes looking at the pre-cleaning data as needed.)

'Sale Price' appear to be more or less normally distributed, with a mean somewhere around $150k.

![png](../images/ames_notebook_files/ames_notebook_28_0.png)

<br>

'Sale Price' varies quite a bit by 'Neighborhood'.

![png](../images/ames_notebook_files/ames_notebook_29_0.png)

<br>

Correlation between 'Sale Price' and 'Lot Area' makes sense, as does an inverse correlation between 'Sale Price' and 'House Age'. Surprisingly, 'Overall Cond' and 'Sale Price' seem barely related.

![png](../images/ames_notebook_files/ames_notebook_30_0.png)

<br>

'1st Fl SF' looks affected by some outliers. '2nd Flr SF' and 'Garage Area' both look affected by the zero values, presumably from houses with only one story or no garage.

![png](../images/ames_notebook_files/ames_notebook_32_0.png)

<br>

'2nd Flr SF' and 'Garage Area' with the zero value removed.

![png](../images/ames_notebook_files/ames_notebook_34_0.png)

<br>

# Modeling and Predictions
---
Predicting 'Sale Price' using three different algorithms, all from Scikit Learn:
- [Random Forest Regressor (from sci-kit learn)](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [AdaBoost Regressor](https://xgboost.readthedocs.io/en/latest/)
- [Support Vector Regressor](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

<br>

### Creating a pipeline to combine standard scaling with the regressor

```python
prf = Pipeline([
    ('ss', StandardScaler()),
    ('rfr', RandomForestRegressor())
])

psv = Pipeline([
    ('ss', StandardScaler()),
    ('svr', SVR())
])

pad = Pipeline([
    ('ss', StandardScaler()),
    ('ada', AdaBoostRegressor())
])

```

<br>

### Splitting the data for validation

```python
X = df_dummies.drop('SalePrice', axis=1)
y = df_dummies['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

<br>

### Initial scores
This project was originally done in the context of a Kaggle competition which used Root Mean Squared Error (RMSE) for scoring, so that is the metric I'm using here.

The function _RMSE_score_ returns a 5x cross-validated RMSE for a given model:

```python
# creating a function to return 5x cross-validated RMSE

def RMSE_score(model, X, y, cv=5):
    return (np.mean(cross_val_score(prf, X_test, y_test, scoring='neg_mean_squared_error', cv=cv))*-1)**0.5
```

<br>

Scoring all three models using _RMSE_score_...

```python
models = [prf, psv, pad]
model_scores = {
                'model':['Random Forest','SVR','AdaBoost'],
                'scores_1':[]
               }
for m in models:
    m.fit(X_train, y_train)
    RMSE = RMSE_score(m, X_test, y_test)
    model_scores['scores_1'].append(RMSE)
    print(f'{m.steps[1][0]} score:')
    print(RMSE)
    print('')
```

<br>

The first round of scores: 

    Random Forest Regressor RMSE:
    29854.4777408
    
    Support Vector Regressor RMSE:
    29697.6673884
    
    AdaBoost Regressor RMSE:
    29424.7745198
    

<br>

A graphical look ... every model returns a score a shade below 30000.

![png](../images/ames_notebook_files/ames_notebook_45_0.png)

<br>

### Improving the data?
Looking at the scatter plots above, a couple of the features look like they could benefit from converting to a log scale.

Here's '1st Fl SF' converted to log ...

![png](../images/ames_notebook_files/ames_notebook_48_0.png)

<br>

And here's 'Lot Area'.

![png](../images/ames_notebook_files/ames_notebook_49_0.png)

<br>

Converting both to log scale and re-scoring models....

```python
X = df_dummies.drop('SalePrice', axis=1)
y = df_dummies['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

```python
model_scores['scores_2'] = []
for m in models:
    m.fit(X_train, y_train)
    RMSE = RMSE_score(m, X_test, y_test)
    model_scores['scores_2'].append(RMSE)
    print(f'{m.steps[1][0]} score:')
    print(RMSE)
    print('')
```

<br>

Did the scores improve?

    Random Forest Regressor RMSE:
    29002.6115943
    
    Support Vector Regressor RMSE:
    29052.4535218
    
    AdaBoost Regressor RMSE:
    28461.5800328
    

<br>

Marginal improvements in all models. (Remember that RMSE measures error, so we are trying to minimize the score.)

![png](../images/ames_notebook_files/ames_notebook_56_0.png)

<br>


# Model Tuning
---
Instead of tweaking the data, let's see if tuning our models brings the score down across the board. Running a few GridSearches to optimize our three. models.

<br>

### RandomForest
Searching for best parameters across 'n_estimators', 'max_features' and 'min_samples_split'.


```python
# set params to search over
params = {
        'rfr__n_estimators':list(range(10,100, 20)),
        'rfr__max_features':[0.5, 'sqrt', 'log2', 'auto'],
        'rfr__min_samples_split':[2, 3, 5, 8]
        }

# instantiate Grid Search
gs_rfr = GridSearchCV(prf, param_grid=params, cv=5, scoring='neg_mean_squared_error')
```


```python
X = df_dummies.drop('SalePrice', axis=1)
y = df_dummies['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# fitting the Grid Search
gs_rfr.fit(X_train, y_train)
```

```python
model_scores['scores_tuned'] = []

RMSE = RMSE_score(gs_rfr, X_test, y_test)
model_scores['scores_tuned'].append(RMSE)
print(RMSE)
```

    tuned Random Forest RMSE:
    29491.6725595

Actually worse than the initial score of around 29000.

<br>

### SVR
Can we improve SVR by using a different kernel and adjusting 'C' and 'gamma'?

```python
# resetting kernel in Pipeline
psv = Pipeline([
    ('ss', StandardScaler()),
    ('svr', SVR(kernel='linear'))
])

# set params to search over
params = {
        'svr__C':[0.001, 0.01, 0.1, 1, 10],
        'svr__gamma':[0.001, 0.01, 0.1, 1]
        }

# instantiate Grid Search
gs_svr = GridSearchCV(psv, param_grid=params, cv=5, scoring='neg_mean_squared_error')
```

```python
X = df_dummies.drop('SalePrice', axis=1)
y = df_dummies['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# fitting the Grid Search
gs_svr.fit(X_train, y_train)
```

```python
# new score?
RMSE = RMSE_score(gs_svr, X_test, y_test)
model_scores['scores_tuned'].append(RMSE)
print(RMSE)
```

    tuned SVR RMSE:   
    34941.5142229

As with Random Forest, no improvement from tuning the model.

<br>

### AdaBoost
Tweaking 'n_estimators', 'loss' and 'learning_rate'.

```python
# set params to search over
params = {
        'ada__n_estimators':[50, 75, 100],
        'ada__loss':['linear', 'square', 'exponential'],
        'ada__learning_rate':[0.5, 1, 1.5]
        }

# instantiate Grid Search
gs_ada = GridSearchCV(pad, param_grid=params, cv=5, scoring='neg_mean_squared_error')
```

```python
X = df_dummies.drop('SalePrice', axis=1)
y = df_dummies['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

```python
gs_ada.fit(X_train, y_train)
```

```python
RMSE = RMSE_score(gs_ada, X_test, y_test)
model_scores['scores_tuned'].append(RMSE)
print(RMSE)
```

    tuned AdaBoost RMSE: 
    31379.466642

<br>

# Results of tuned models
---
Tuning actually made all three models worse.

![png](../images/ames_notebook_files/ames_notebook_86_0.png)

<br>

# Interpretation and Conclusion
---
The benefits of model tuning entirely depend upon picking the right parameters to adjust over the right range of values. In this case, I probably could have chosen better.
