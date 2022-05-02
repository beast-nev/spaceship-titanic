import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras as keras
import keras.layers as layers
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
x_train = x_train.drop(columns=["Transported", "PassengerId"])
x_test = x_test.drop(columns=["PassengerId"])

categorical_features = ["HomePlanet", "CryoSleep",
                        "Cabin", "Destination", "VIP", "Name"]
float_features = ["Age", "RoomService",
                  "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

for i in float_features:
    if i != "Age":
        x_train[i] = x_train[i].fillna(0)
        x_test[i] = x_test[i].fillna(0)
    else:
        x_train[i] = SimpleImputer(strategy="mean").fit_transform(x_train[[i]])
        x_test[i] = SimpleImputer(strategy="mean").fit_transform(x_test[[i]])


x_train["total_spent"] = x_train["RoomService"] + \
    x_train["FoodCourt"] + x_train["ShoppingMall"] + \
    x_train["Spa"] + x_train["VRDeck"]

x_test["total_spent"] = x_test["RoomService"] + \
    x_test["FoodCourt"] + x_test["ShoppingMall"] + \
    x_test["Spa"] + x_test["VRDeck"]

# deck/num/side -> Side = P or S
# B/0/P
x_train[['deck', 'num', 'side']] = x_train['Cabin'].str.split('/', expand=True)

x_train = x_train.drop(columns=["Cabin", ])

x_test[['deck', 'num', 'side']] = x_test['Cabin'].str.split('/', expand=True)

x_test = x_test.drop(columns=["Cabin", ])

deckEncoder = LabelEncoder()
x_train["deck"] = deckEncoder.fit_transform(x_train["deck"])
x_test["deck"] = deckEncoder.fit_transform(x_test["deck"])

numEncoder = LabelEncoder()
x_train["num"] = deckEncoder.fit_transform(x_train["num"])
x_test["num"] = deckEncoder.fit_transform(x_test["num"])

portEncoder = OneHotEncoder()
x_train["side"] = deckEncoder.fit_transform(x_train["side"])
x_test["side"] = deckEncoder.fit_transform(x_test["side"])

feature_names = x_train.columns
print("Number of features: ", len(feature_names))

x_train['total_spent'] = x_train.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['total_spent'],
    axis=1
)

x_test['total_spent'] = x_test.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['total_spent'],
    axis=1
)

for i in categorical_features:
    if i != "Cabin" and i != "Name":
        encoder = OneHotEncoder()
        x_train[i] = encoder.fit_transform(x_train[[i]]).toarray()
        x_test[i] = encoder.fit_transform(x_test[[i]]).toarray()
    elif i == "Name":
        encoder = LabelEncoder()
        encoder.fit(x_train[i])
        x_train[i] = encoder.fit_transform(x_train[[i]])
        x_test[i] = encoder.fit_transform(x_test[[i]])

feature_names = x_train.columns

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

input_shape = X_train.shape[0]

model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(units=256, input_shape=[input_shape], activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(units=256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(units=256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(units=256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(units=1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"]
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    x=X_train, y=Y_train,
    validation_data=(X_test, Y_test),
    batch_size=128,
    epochs=500,
    callbacks=[early_stopping]
)

history_df = pd.DataFrame(history.history)
# print(history_df.columns)
history_df.loc[:, ['loss', 'val_loss', ]].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
# print("Maximum binning accuracy: {}".format(
#     history_df['binary_accuracy'].max()))

pred_val = model.predict(X_test)

print("ROC_AUC: ", roc_auc_score(Y_test, pred_val))

pred = model.predict(x_test)

submission["Transported"] = np.round(pred)
submission["Transported"].loc[submission["Transported"] == 1] = True
submission["Transported"].loc[submission["Transported"] == 0] = False

os.makedirs('submissions/nn', exist_ok=True)
submission.to_csv('submissions/nn/out.csv', index=False)
plt.show()
