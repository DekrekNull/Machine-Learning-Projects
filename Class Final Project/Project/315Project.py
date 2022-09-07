from random import random
import numpy as np
import pandas as pd

# Sources:
# https://pandas.pydata.org/docs/reference/frame.html
# https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas/
# https://www.bluegranite.com/blog/predicting-sales-with-the-aid-of-pandas


def print_stats(data_frame):
    print(data_frame.shape)
    print(list(data_frame.columns))
    print(data_frame.isnull().any(axis=0))
    print(data_frame.info())

def condition_to_int(cond):
    # Ran the bellow statement once to get the following dictionary of possible conditions
    # print(data_frame.condition.unique())
    condition = { 'good': 2, 'excellent': 3, 'fair': 1, 'like new': 4, 'new': 5, 'salvage': 0 }
    if (cond is not np.nan):
        cond = condition[cond]
    return cond

def cylinder_count_to_int(cyl):
    # Ran the bellow statement once to get the following dictionary for the cylinder counts

    cylinders = {'8 cylinders': 8, '6 cylinders': 6,  '4 cylinders':4, '5 cylinders':5,
                 '10 cylinders': 10, '3 cylinders':3, 'other':np.nan, '12 cylinders':12 }
    if (cyl is not np.nan):
        cyl = cylinders[cyl]
    return cyl

def year_to_post_age(year):
    year = int(year[:4])
    year = 2022 - year
    return year

# Determines if a row is bad
# - if it is missing more than 4 fields
# - if it has 0 or negative price/odometer reading - unknown meaning makes them bad values and may sway results too much
def bad_row(row):

    if row.isnull().sum() > 4:
        return True
    
    if row.price <= 0 or row.odometer < 0:
        return True

    return False

# Based the following function based on one I found in the bellow project that used the same dataset
# https://www.kaggle.com/code/jerrymazeyu/predict-car-price-by-catboost/notebook
# data_frame - Pandas.DataFrame structure to perform the fill on
# column - The column to fill nan values in
# based_column - The column to use to determine the values to use to fill
# mode - mode used to find fill value: mean, median, or mode
# default - default value to use in the event that a value isn't found
def fill_col_empties(data_frame, column, based_column, mode='mode', default='other'):
    # groups the data frame based on the based_column values, then finds the mode value of the column value for each group
    # the dict key is the based_column value of the group and the key is the mode value of the column
    # these values are then used to fill the empty spaces
    if mode == 'mean':
        d = dict(data_frame.groupby(based_column)[column].mean())
    if mode == 'median':
        d = dict(data_frame.groupby(based_column)[column].median(numeric_only=False))
    if mode == 'mode':
        d = dict(data_frame.groupby(based_column)[column].agg(lambda x: pd.Series.mode(x)))
        for (key, value) in d.items():
            if str(value).find('[') != -1:
                d[key] = default
    # Once the dictionary is found, use it to fill in the emtpy spaces of data_frame[column]
    data_frame[column] = data_frame[column].fillna(data_frame[based_column].apply(lambda x: d.get(x)))
    # After the empty spaces are filled through this method, drop any rows that are still empty in this column
    data_frame.drop(data_frame[data_frame[column].isna()].index, inplace=True)

# Goes through the data frame one column at a time and fills the empty spots based on the other values in the column
def fill_all_empty(data_frame):
    # Mean -    best determined through averaging: 
    #           Condition(year), odometer(condition)
    fill_col_empties(data_frame, 'condition', 'year', 'mean', 2)
    fill_col_empties(data_frame, 'odometer', 'year', 'mean',  10000000)

    # Median -  Mode and mean don't suit these well for these so best to leave it to median:
    #           Model(manufacturer), title_status(condition), paint_color(manufacturer)
    #           Median doesn't want to work on non-numbers so unless I find another method these will just use mode
    fill_col_empties(data_frame, 'model', 'manufacturer', 'mode')
    fill_col_empties(data_frame, 'title_status', 'condition', 'mode', 'clean')
    fill_col_empties(data_frame, 'paint_color', 'manufacturer', 'mode', 'white')

    # Mode -    These are all pretty commonly similar for each model so mode should work best:
    #           fuel(model), drive(model), cylinders(model), transmission(model), size(model), type(model)
    fill_col_empties(data_frame, 'fuel', 'model', default='gas')
    fill_col_empties(data_frame, 'drive', 'model', default='2wd')
    fill_col_empties(data_frame, 'cylinders', 'model', default=4)
    fill_col_empties(data_frame, 'transmission', 'model', default='auto')
    fill_col_empties(data_frame, 'size', 'model', default='full-size')
    fill_col_empties(data_frame, 'type', 'model', default='sdn')

    # Make sure these are ints
    data_frame.condition = data_frame.condition.apply(lambda x: int(x))
    data_frame.odometer = data_frame.odometer.apply(lambda x: int(x))
    data_frame.cylinders = data_frame.cylinders.apply(lambda x: int(x))
    data_frame.year = data_frame.year.apply(lambda x: int(x))

def print_samples(data_frame):
    # Print some random samples to get a good idea of how the data looks and how the sampling is doing
    for i in range(0,10):
        data_sample = data_frame.sample(5, replace = True)
        print(data_sample.head())

def train_data(data_frame):
    from sklearn.model_selection import train_test_split
    from sklearn import metrics, svm
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    categorical_features = data_frame.select_dtypes(include=[object])
    numerical_features = data_frame.select_dtypes(include=[np.number])
    #numerical_features = numerical_features.sample(50000, replace=True)
    # Creates new columns for the non-number columns with binary values
    data_dummies = data_frame
    
    # These columns are too diverse and cause the mem use to be way too high, especially model.
    data_dummies.drop(['model', 'size', 'paint_color', 'state'], axis=1, inplace=True)

    data_dummies = pd.get_dummies(data_frame, 
        columns=[ 'manufacturer', 'fuel', 'title_status', 'transmission', 'drive', 'type' ])

    # numerical_features = numerical_features.sample(100000,replace=True)

    # Split into train data and test data
    # train_data, test_data, train_target, test_target = train_test_split(data_dummies.loc[:, [x for x in list(data_dummies.columns) if x not in ['price', 'id']]], data_dummies.loc[:, 'price'], test_size=0.2)
    train_data, test_data, train_target, test_target = train_test_split(numerical_features.loc[:, [x for x in list(numerical_features.columns) if x not in ['price', 'id']]], numerical_features.loc[:, 'price'], test_size=0.2)

     
    # Print the shapes of the train and test data to verify they make sense
    # print("Train")
    # print(train_data.shape)
    # print(train_target.shape)
    # print(train_data.head())
    # print(train_target.head())
    # print("Test")
    # print(test_data.shape)
    # print(test_target.shape)
    # print(test_data.head())
    # print(test_target.head()) 
    

    quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    X_train_trans = quantile_transformer.fit_transform(train_data)
    X_test_trans = quantile_transformer.fit_transform(test_data)

    # I simply just changed this line to use the different algorithms
    model = RandomForestRegressor()
    
    print("Training...")
    model.fit(X_train_trans, train_target)
    predictions = model.predict(test_data)

    print("RMSE: ", np.sqrt(metrics.mean_squared_error(test_target, predictions)))
    print("R2: ", metrics.r2_score(test_target, predictions))
    print("MAE: ", metrics.mean_absolute_error(test_target, predictions))
    print('Score: ', model.score(X_test_trans, test_target))

def main():
    data_frame = pd.read_csv('./vehicles.csv')

    # Drop rows that are missing fields that I think would be too difficult to accurately fill
    data_frame = data_frame.dropna(axis=0, subset=[ 'year', 'manufacturer', 'posting_date'])

    # Drop rows that qualify as bad based on conditions outlined in bad_row()
    data_frame['bad_row'] = data_frame.apply(bad_row, axis=1)
    data_frame.drop(data_frame[data_frame.bad_row].index, inplace=True)
    # Column no longer needed so we can drop it
    data_frame.drop(['bad_row'], axis=1, inplace=True)

    # Converts each condition string to its respective int value to take the mean easier
    data_frame['condition'] = data_frame['condition'].apply(condition_to_int)
    data_frame['cylinders'] = data_frame['cylinders'].apply(cylinder_count_to_int)
    data_frame['post_age'] = data_frame['posting_date'].apply(year_to_post_age)
    data_frame.drop(['posting_date'], axis=1, inplace=True)

    # Now that the unused rows are dropped, we need to fill the remaining empty spots
    fill_all_empty(data_frame)

    #print_samples(data_frame)

    # Now that we have confirmed our dataset contains no empty spots, we can split and train/test!
    train_data(data_frame)

    

if __name__ == '__main__':
	main()