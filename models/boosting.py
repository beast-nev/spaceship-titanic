import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from pandas.plotting import autocorrelation_plot


x_train = pd.read_csv('./data/train.csv')
x_test = pd.read_csv('./data/test.csv')

submission = pd.DataFrame(
    columns=["PassengerId", "Transported"], data=x_test["PassengerId"])

y_train = x_train["Transported"]
x_train = x_train.drop(columns=["Transported", "PassengerId"])
x_test = x_test.drop(columns=["PassengerId"])

categorical_features = ["HomePlanet", "CryoSleep",
                        "Cabin", "Destination", "VIP", "Name"]
float_features = ["Age", "RoomService",
                  "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

encoder = LabelEncoder()

for i in categorical_features:
    x_train[i] = encoder.fit_transform(x_train[i])
    x_test[i] = encoder.fit_transform(x_test[i])

feature_names = x_train.columns

imputer = SimpleImputer()
x_train = imputer.fit_transform(x_train)
x_test = imputer.fit_transform(x_test)


# print("Count of True in training: ", y_train.loc[y_train == True])
# 4378
# print("Count of False in training: ", y_train.loc[y_train == False])
# 4315

# # Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
x_train = pd.DataFrame(data=x_train, columns=feature_names)
x_test = pd.DataFrame(data=x_test, columns=feature_names)

scaler = MinMaxScaler()
for i in float_features:
    # print("Column:", i, " with data before scaling:", x_train[i])
    x_train[i] = scaler.fit_transform(x_train[[i]])
    # print("Column:", i, " with data after scaling:", x_train[i])
    x_test[i] = scaler.fit_transform(x_test[[i]])

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train, test_size=0.33, random_state=42)

model = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.5, max_depth=5, min_samples_split=2, min_samples_leaf=1)

# sfs = SequentialFeatureSelector(
#     model, n_features_to_select=6, direction="forward", n_jobs=-1)
# sfs.fit(X_train, Y_train)

# mask = sfs.get_support()
# features_chosen_mask = feature_names[mask]
# features_chosen = [feature
#                    for feature in features_chosen_mask]
# print("Features chosen: ", features_chosen)
# 'CryoSleep', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' <- FSS
# 'Destination', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' <- BSS

selected_features_FSS = ['CryoSleep', 'RoomService',
                         'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# X_train = sfs.transform(X_train)
# X_test = sfs.transform(X_test)
# x_test = sfs.transform(x_test)

X_train = X_train[selected_features_FSS]
X_test = X_test[selected_features_FSS]
x_test = x_test[selected_features_FSS]

model.fit(X_train, Y_train)

print("Score: ",  model.score(X_test, Y_test))

pred = model.predict(x_test)
submission["Transported"] = pred

os.makedirs('submissions/boosting', exist_ok=True)
submission.to_csv('submissions/boosting/out.csv', index=False)
