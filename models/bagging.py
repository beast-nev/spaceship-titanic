import os
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
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
    if i != "Cabin":
        x_train[i] = encoder.fit_transform(x_train[i])
        x_test[i] = encoder.fit_transform(x_test[i])

x_train["total_spent"] = x_train["RoomService"] + \
    x_train["FoodCourt"] + x_train["ShoppingMall"] + \
    x_train["Spa"] + x_train["VRDeck"]

x_train["interactionAmentities1"] = x_train["FoodCourt"] * \
    x_train["ShoppingMall"]
x_train["interactionAmentities2"] = x_train["FoodCourt"] * \
    x_train["RoomService"]
x_train["interactionAmentities3"] = x_train["FoodCourt"] * \
    x_train["Spa"]
x_train["interactionAmentities4"] = x_train["FoodCourt"] * \
    x_train["VRDeck"]


# deck/num/side -> Side = P or S
# B/0/P
x_train["Cabin"] = x_train["Cabin"].str.replace('/', '')
x_train["deck"] = x_train["Cabin"].str[0:1]
x_train["num"] = x_train["Cabin"].str[1:2]
x_train["side"] = x_train["Cabin"].str[2:]
x_train["side"] = x_train["side"].str.replace('0', '')
x_train = x_train.drop(columns=["Cabin"])

x_test["Cabin"] = x_test["Cabin"].str.replace('/', '')
x_test["deck"] = x_test["Cabin"].str[0:1]
x_test["num"] = x_test["Cabin"].str[1:2]
x_test["side"] = x_test["Cabin"].str[2:]
x_test["side"] = x_test["side"].str.replace('0', '')
x_test = x_test.drop(columns=["Cabin"])

deckEncoder = LabelEncoder()
x_train["deck"] = deckEncoder.fit_transform(x_train["deck"])
x_test["deck"] = deckEncoder.fit_transform(x_test["deck"])

numEncoder = LabelEncoder()
x_train["num"] = deckEncoder.fit_transform(x_train["num"])
x_test["num"] = deckEncoder.fit_transform(x_test["num"])

portEncoder = LabelEncoder()
x_train["side"] = deckEncoder.fit_transform(x_train["side"])
x_test["side"] = deckEncoder.fit_transform(x_test["side"])

x_test["total_spent"] = x_test["RoomService"] + \
    x_test["FoodCourt"] + x_test["ShoppingMall"] + \
    x_test["Spa"] + x_test["VRDeck"]

x_test["interactionAmentities1"] = x_test["FoodCourt"] * \
    x_test["ShoppingMall"]
x_test["interactionAmentities2"] = x_test["FoodCourt"] * \
    x_test["RoomService"]
x_test["interactionAmentities3"] = x_test["FoodCourt"] * \
    x_test["Spa"]
x_test["interactionAmentities4"] = x_test["FoodCourt"] * \
    x_test["VRDeck"]

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

print("Number of features:", len(feature_names))

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42)

model = BaggingClassifier(n_estimators=250, n_jobs=-1,
                          base_estimator=KNeighborsClassifier(n_neighbors=5))

selected_features_FSS = ['CryoSleep', 'Age', 'RoomService',
                         'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'total_spent', 'deck']

X_train = X_train[selected_features_FSS]
X_test = X_test[selected_features_FSS]
x_test = x_test[selected_features_FSS]

model.fit(X_train, Y_train)

y_pred_val = model.predict(X_test)

print("Score: ",  model.score(X_test, Y_test))
print("Roc auc: ",  roc_auc_score(Y_test, y_pred_val))

pred = model.predict(x_test)
submission["Transported"] = pred

os.makedirs('submissions/bagging', exist_ok=True)
submission.to_csv('submissions/bagging/out.csv', index=False)
