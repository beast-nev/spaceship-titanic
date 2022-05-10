import os
from time import asctime, localtime, time
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
        x_train[i] = SimpleImputer(
            strategy="mean").fit_transform(x_train[[i]])
        x_test[i] = SimpleImputer(
            strategy="mean").fit_transform(x_test[[i]])

x_train["most_spent"] = x_train[["RoomService",
                                 "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].max(axis=1)
x_train["least_spent"] = x_train[["RoomService",
                                 "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].min(axis=1)
x_train["std_spent"] = x_train[["RoomService",
                                "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].std(axis=1)
x_train["total_spent"] = x_train[["RoomService",
                                 "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)

x_test["most_spent"] = x_test[["RoomService",
                               "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].max(axis=1)
x_test["least_spent"] = x_test[["RoomService",
                                "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].min(axis=1)
x_test["std_spent"] = x_test[["RoomService",
                              "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].std(axis=1)
x_test["total_spent"] = x_test[["RoomService",
                                "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)

# # deck/num/side -> Side = P or S
# # B/0/P
x_train[['deck', 'num', 'side']] = x_train['Cabin'].str.split('/', expand=True)

x_train = x_train.drop(columns=["Cabin", ])

x_test[['deck', 'num', 'side']] = x_test['Cabin'].str.split('/', expand=True)

x_test = x_test.drop(columns=["Cabin", ])

x_train["deck"] = LabelEncoder().fit_transform(x_train["deck"])
x_test["deck"] = LabelEncoder().fit_transform(x_test["deck"])

x_train["num"] = LabelEncoder().fit_transform(x_train["num"])
x_test["num"] = LabelEncoder().fit_transform(x_test["num"])

x_train["side"] = OneHotEncoder(sparse=False).fit_transform(x_train[["side"]])
x_test["side"] = OneHotEncoder(sparse=False).fit_transform(x_test[["side"]])

feature_names = x_train.columns
print("Number of features: ", len(feature_names))

x_train['total_spent'] = x_train.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['total_spent'],
    axis=1
)
x_train['most_spent'] = x_train.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['most_spent'],
    axis=1
)
x_train['least_spent'] = x_train.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['least_spent'],
    axis=1
)
x_train['std_spent'] = x_train.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['std_spent'],
    axis=1
)

x_test['total_spent'] = x_test.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['total_spent'],
    axis=1
)
x_test['most_spent'] = x_test.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['most_spent'],
    axis=1
)
x_test['least_spent'] = x_test.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['least_spent'],
    axis=1
)
x_test['std_spent'] = x_test.apply(
    lambda row: 0 if row["CryoSleep"] == True or row["Age"] <= 12 else row['std_spent'],
    axis=1
)

for i in categorical_features:
    if i != "Cabin" and i != "CryoSleep" and i != "VIP":
        x_train[i] = x_train[i].fillna("")
        x_test[i] = x_test[i].fillna("")
    elif i == "CryoSleep" or i == "VIP":
        x_train[i] = x_train[i].fillna(False)
        x_test[i] = x_test[i].fillna(False)

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

input_shape = X_train.shape[1]
layer_sizes = [756, 756, 512, 256, 128, 1]
activation_function = "relu"
dropout_size = 0.5
model = keras.Sequential([
    layers.BatchNormalization(input_shape=[input_shape]),
    layers.Dense(input_shape=[input_shape], units=layer_sizes[0], activation=activation_function,
                 ),
    layers.BatchNormalization(),
    layers.Dropout(dropout_size),
    layers.Dense(units=layer_sizes[1], activation=activation_function,
                 ),
    layers.BatchNormalization(),
    layers.Dropout(dropout_size),
    layers.Dense(units=layer_sizes[len(layer_sizes)-1], activation="sigmoid"),
])

# compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"]
)

# set up early_stopping to avoid overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=50,
    min_delta=0.001,
    restore_best_weights=True,
)
# fit the model
batch_size = 128
epochs = 500
history = model.fit(
    x=X_train, y=Y_train,
    validation_data=(X_test, Y_test),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
)

# plot the trainiing loss versus validation loss to confirm no under/overfitting
history_df = pd.DataFrame(history.history)
print(history_df.columns)
history_df.loc[:, ['loss', 'val_loss', ]].plot()
print(f"Minimum validation loss: {history_df['val_loss'].min()}")
print(f"Best val accuracy: {history_df['val_binary_accuracy'].max()}")

# write results to file for evaluating model
with open("run_results.txt", "a") as file:
    file.write(f"\nTime: {asctime(localtime())}\n")
    file.write(
        f"Val binary accuracy best: {str(history_df['val_binary_accuracy'].max())}\n")
    file.write(
        f"With layer sizes: {layer_sizes},\nbatch_size: {batch_size},\nepochs: {epochs},\nactivation function: {activation_function},\ndropout size: {dropout_size}\n")
pred = model.predict(x_test)
# pred = stats.rankdata(pred)

submission["Transported"] = np.round(pred)
submission["Transported"].loc[submission["Transported"] == 1] = True
submission["Transported"].loc[submission["Transported"] == 0] = False

os.makedirs('submissions/nn', exist_ok=True)
submission.to_csv('submissions/nn/out.csv', index=False)
plt.show()
