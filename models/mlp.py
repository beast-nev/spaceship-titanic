import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split

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

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train, test_size=0.33, random_state=42)

model = MLPClassifier(max_iter=1000, activation="logistic", batch_size=500, hidden_layer_sizes=(
    6,), learning_rate="adaptive", learning_rate_init=0.001, solver="adam")

sfs = SequentialFeatureSelector(
    model, n_features_to_select=6, direction="forward", n_jobs=-1)
sfs.fit(X_train, Y_train)

mask = sfs.get_support()
features_chosen_mask = feature_names[mask]
features_chosen = [feature
                   for feature in features_chosen_mask]
print("Features chosen: ", features_chosen)
# Score:  0.7141861275705821 <- BSS

X_train = sfs.transform(X_train)
X_test = sfs.transform(X_test)
x_test = sfs.transform(x_test)

# 'activation': 'logistic', 'batch_size': 500, 'hidden_layer_sizes': (6,), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'solver': 'adam'
# mlp_space = {
#     'hidden_layer_sizes': [(12, 36, 12), (12,), (6,), (10,), (8,)],
#     'activation': ["relu", "logistic"],
#     'solver': ["adam", "sgd"],
#     'learning_rate': ['constant', 'adaptive'],
#     'batch_size': [200, 500],
#     'learning_rate_init': [0.001, 0.05, 0.0001],
# }
# grid_search = GridSearchCV(model, mlp_space, n_jobs=-1, cv=5)
# grid_search.fit(x_train, y_train)
# print("Best estimator: ", grid_search.best_estimator_)
# print("Best params: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)
# 'activation': 'logistic', 'batch_size': 200, 'hidden_layer_sizes': (12, 36, 12), 'learning_rate': 'constant', 'learning_rate_init': 0.001, 'solver': 'adam'
model.fit(X_train, Y_train)

score = model.score(X_test, Y_test)

print("Score: ", score)

pred = model.predict(x_test)
submission["Transported"] = pred

os.makedirs('submissions/mlp', exist_ok=True)
submission.to_csv('submissions/mlp/out.csv', index=False)
