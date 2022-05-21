import os
from re import L
from time import asctime, localtime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgbm

pd.set_option('display.max_columns', 25)
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

x_train = pd.read_csv('./data/train.csv')
x_test = pd.read_csv('./data/test.csv')

submission = pd.DataFrame(
    columns=["PassengerId", "Transported"], data=x_test["PassengerId"])

y_train = x_train["Transported"]
x_train = x_train.drop(columns=["Transported", ])

float_features = ["Age", "RoomService",
                  "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "most_spent", "least_spent", "std_spent", "total_spent"]

label_encoders = ["FirstName",
                  "LastName",
                  "num", "GroupId"]
onehot_encoders = ["HomePlanet", "CryoSleep",
                   "deck", "side", "Destination", "VIP"]


def fill_nulls(df):

    # fill the null values with the mean, except for age -> set to 0
    for i in float_features:
        if i != "Age":
            df[i] = df[i].fillna(0)
        else:
            df[i] = SimpleImputer(
                strategy="mean").fit_transform(df[[i]])

    # label encoding and one hot encoding
    for j in label_encoders:
        df[j] = LabelEncoder().fit_transform(df[j])
    for k in onehot_encoders:
        df[k] = OneHotEncoder().fit_transform(df[[i]]).toarray()
    return df


def feature_engineering(df):

    # calculate the most, least, std, and total spent for each person
    df["most_spent"] = df[["RoomService",
                           "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].max(axis=1)
    df["least_spent"] = df[["RoomService",
                            "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].min(axis=1)
    df["std_spent"] = df[["RoomService",
                          "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].std(axis=1)
    df["total_spent"] = df[["RoomService",
                            "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)

    # split the cabin into three features
    df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(columns=["Cabin", ])

    # if the person is sleeping or less than 12, make the total spend amounts 0
    df['total_spent'] = df.apply(
        lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['total_spent'],
        axis=1
    )
    df['most_spent'] = df.apply(
        lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['most_spent'],
        axis=1
    )
    df['least_spent'] = df.apply(
        lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['least_spent'],
        axis=1
    )
    df['std_spent'] = df.apply(
        lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['std_spent'],
        axis=1
    )

    # split name into first and last name
    df['FirstName'] = df['Name'].str.split(' ', expand=True)[0]
    df['LastName'] = df['Name'].str.split(' ', expand=True)[1]
    df.drop(columns=['Name'], inplace=True)

    # split travelers into groups based on their id
    df['GroupId'] = df['PassengerId'].str.split('_', expand=True)[
        0]
    return df


# transform the training and test data
x_train = feature_engineering(x_train)
x_train = fill_nulls(x_train)
x_train = x_train.drop(columns=['PassengerId'])

x_test = feature_engineering(x_test)
x_test = fill_nulls(x_test)
x_test = x_test.drop(columns=['PassengerId'])

# number of features
feature_names = x_train.columns
print("Number of features: ", len(feature_names))

y_preds = []


# From https://www.kaggle.com/code/kartik2khandelwal/ensemble-model-xgboost-catboost-lgbm/notebook?scriptVersionId=90693221
lgbm_param = {'boosting_type': 'gbdt',
              'lambda_l1': 0.134746148489252,
              'lambda_l2': 0.10521615726990495,
              'bagging_fraction': 0.7,
              'feature_fraction': 0.6,
              'learning_rate': 0.001019958066572347,
              'max_depth': 9,
              'num_leaves': 23,
              'min_child_samples': 46}

skfold = StratifiedKFold(n_splits=10)
for fold, (train_id, test_id) in enumerate(skfold.split(x_train, y_train)):

    # split into the folds
    X_train = x_train.iloc[train_id]
    Y_train = y_train.iloc[train_id]
    X_test = x_train.iloc[test_id]
    Y_test = y_train.iloc[test_id]

    X_train = np.asarray(X_train).astype('float32')
    X_test = np.asarray(X_test).astype('float32')
    Y_train = np.asarray(Y_train).astype('float32')
    Y_test = np.asarray(Y_test).astype('float32')

    # # run the model on the fold
    lgbm_model = lgbm.LGBMClassifier(**lgbm_param)
    lgbm_model.fit(X_train, Y_train)
    print(f"Model score: {lgbm_model.score(X_test, Y_test)}")
    pred = lgbm_model.predict(x_test)
    y_preds.append(pred)

pred = sum(y_preds) / len(y_preds)
submission['Transported'] = pred
submission['Transported'] = np.where(
    submission['Transported'] > 0.5, True, False)

os.makedirs('submissions/boosting', exist_ok=True)
submission.to_csv('submissions/boosting/out.csv', index=False)
plt.show()
