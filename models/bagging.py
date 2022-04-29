import os
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
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
                  "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "total_spent"]

encoder = LabelEncoder()
for i in categorical_features:
    x_train[i] = encoder.fit_transform(x_train[i])
    x_test[i] = encoder.fit_transform(x_test[i])

x_train["total_spent"] = x_train["RoomService"] + \
    x_train["FoodCourt"] + x_train["ShoppingMall"] + \
    x_train["Spa"] + x_train["VRDeck"]
x_test["total_spent"] = x_test["RoomService"] + \
    x_test["FoodCourt"] + x_test["ShoppingMall"] + \
    x_test["Spa"] + x_test["VRDeck"]

feature_names = x_train.columns

imputer = SimpleImputer()
x_train = imputer.fit_transform(x_train)
x_test = imputer.fit_transform(x_test)

x_train = pd.DataFrame(data=x_train, columns=feature_names)
x_test = pd.DataFrame(data=x_test, columns=feature_names)

scaler = StandardScaler()
for i in float_features:
    # if i != "Age":
    # print("Column:", i, " with data before scaling:", x_train[i])
    x_train[i] = scaler.fit_transform(x_train[[i]])
    # print("Column:", i, " with data after scaling:", x_train[i])
    x_test[i] = scaler.fit_transform(x_test[[i]])

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42)

selected_features_FSS = ['CryoSleep', 'RoomService',
                         'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', "total_spent"]

X_train = X_train[selected_features_FSS]
X_test = X_test[selected_features_FSS]
x_test = x_test[selected_features_FSS]

model = BaggingClassifier(n_estimators=500, n_jobs=-1,
                          base_estimator=KNeighborsClassifier(n_neighbors=9))
model.fit(X_train, Y_train)

print("Score: ",  model.score(X_test, Y_test))

pred = model.predict(x_test)
submission["Transported"] = pred

os.makedirs('submissions/random_forests', exist_ok=True)
submission.to_csv('submissions/random_forests/out.csv', index=False)
