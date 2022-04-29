import os
from re import X
from tkinter import Grid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, chi2, f_classif, mutual_info_classif
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFdr, SelectPercentile, SelectFwe


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

# print("Number of VIP in training: ",
#       x_train["VIP"].loc[x_train["VIP"] == True].shape[0])
# print("Number of NON-VIP in training: ",
#       x_train["VIP"].loc[x_train["VIP"] == False].shape[0])
# print("Number of VIP in testing: ",
#       x_test["VIP"].loc[x_test["VIP"] == True].shape[0])
# print("Number of NON-VIP in testing: ",
#       x_test["VIP"].loc[x_test["VIP"] == False].shape[0])

# print("Number of cryoSleep in training: ",
#       x_train["CryoSleep"].loc[x_train["CryoSleep"] == True].shape[0])
# print("Number of NON-cryoSleep in training: ",
#       x_train["CryoSleep"].loc[x_train["CryoSleep"] == False].shape[0])
# print("Number of cryoSleep in testing: ",
#       x_test["CryoSleep"].loc[x_test["CryoSleep"] == True].shape[0])
# print("Number of NON-cryoSleep in testing: ",
#       x_test["CryoSleep"].loc[x_test["CryoSleep"] == False].shape[0])

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
print("Number of features: ", len(feature_names))

imputer = SimpleImputer()
x_train = imputer.fit_transform(x_train)
x_test = imputer.fit_transform(x_test)

x_train = pd.DataFrame(data=x_train, columns=feature_names)
x_test = pd.DataFrame(data=x_test, columns=feature_names)

# x_train_float = x_train[float_features]

# scaler = StandardScaler()
# for i in float_features:
#     x_train[i] = scaler.fit_transform(x_train[[i]])
#     x_test[i] = scaler.fit_transform(x_test[[i]])

# x_train = x_train.drop(columns=["Spa", "VRDeck", "ShoppingMall"])
# x_test = x_test.drop(columns=["Spa", "VRDeck", "ShoppingMall"])

feature_names = x_train.columns

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42)

# sfs = SequentialFeatureSelector(
#     model, n_features_to_select=10, direction="forward", n_jobs=-1)
# sfs.fit(X_train, Y_train)

# mask = sfs.get_support()
# features_chosen_mask = feature_names[mask]
# features_chosen = [feature
#                    for feature in features_chosen_mask]
# print("Features chosen: ", features_chosen)

# X_train = sfs.transform(X_train)
# X_test = sfs.transform(X_test)
# x_test = sfs.transform(x_test)

# selector = SelectKBest(k=14, score_func=f_classif)
# print("SelectKBest with k=", selector.get_params()["k"])
# selector.fit(X_train, Y_train)

# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)
# x_test = selector.transform(x_test)

selected_features_FSS = ['CryoSleep', 'RoomService',
                         'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

X_train = X_train[selected_features_FSS]
X_test = X_test[selected_features_FSS]
x_test = x_test[selected_features_FSS]

model = RandomForestClassifier(class_weight='balanced', criterion='gini',
                               max_depth=7, max_features='sqrt', max_leaf_nodes=30,
                               n_estimators=350, n_jobs=-1, random_state=42)

model.fit(X_train, Y_train)

pred_val = model.predict(X_test)

print("Score: ",  model.score(X_test, Y_test))
print("ROC_AUC: ", roc_auc_score(Y_test, pred_val))
print("Classification report: ", classification_report(Y_test, pred_val))

pred = model.predict(x_test)

submission["Transported"] = pred

os.makedirs('submissions/random_forests', exist_ok=True)
submission.to_csv('submissions/random_forests/out.csv', index=False)
