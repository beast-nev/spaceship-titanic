import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split

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

x_train = pd.DataFrame(data=x_train, columns=feature_names)
x_test = pd.DataFrame(data=x_test, columns=feature_names)

scaler = RobustScaler()
for i in float_features:
    if i != "Age":
        # print("Column:", i, " with data before scaling:", x_train[i])
        x_train[i] = scaler.fit_transform(x_train[[i]])
        # print("Column:", i, " with data after scaling:", x_train[i])
        x_test[i] = scaler.fit_transform(x_test[[i]])

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42)

# n_estimators = 500, max_depth = 5 <- Best on Kaggle rn

selected_features_FSS = ['CryoSleep', 'RoomService',
                         'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

X_train = X_train[selected_features_FSS]
X_test = X_test[selected_features_FSS]
x_test = x_test[selected_features_FSS]

print(y_train)
model = RidgeClassifier(class_weight={0: 0.5, 1: 0.73})

model.fit(X_train, Y_train)

y_pred = pd.DataFrame(data=model.predict(
    X_test), columns=["pred"])
print("Score: ",  model.score(X_test, Y_test))
print("Number of True in Y_Test: ",
      y_pred["pred"].loc[y_pred["pred"] == True].shape[0])
print("Number of False in Y_Test: ",
      y_pred["pred"].loc[y_pred["pred"] == False].shape[0])

pred = model.predict(x_test)
submission["Transported"] = pred

os.makedirs('submissions/ridge', exist_ok=True)
submission.to_csv('submissions/ridge/out.csv', index=False)

# best_rf
# True = 2338
# False = 1939
# 1.2

# 0.76876
# Number of True in Y_Test:  1259
# Number of False in Y_Test:  1349

# 0.76665
# Number of True in Y_Test:  1373
# Number of False in Y_Test:  1235
