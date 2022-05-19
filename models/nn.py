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
x_train = x_train.drop(columns=["Transported", ])
# x_test = x_test.drop(columns=["PassengerId"])

float_features = ["Age", "RoomService",
                  "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "most_spent", "least_spent", "std_spent", "total_spent"]

label_encoders = ["FirstName",
                  "LastName",
                  "num", ]
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

# split into train and validation set
X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

# convert to float32 for tf
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
Y_train = np.asarray(Y_train).astype('float32')
Y_test = np.asarray(Y_test).astype('float32')

# nn model
input_shape = X_train.shape[1]
layer_sizes = [1024, 756, 512, 256, 128, 1]
activation_function = "swish"
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
x_test = np.asarray(x_test).astype('float32')
pred = model.predict(x_test)
# pred = stats.rankdata(pred)

submission['Transported'] = np.array(pred).mean(axis=1)
submission['Transported'] = np.where(
    submission['Transported'] > 0.5, True, False)

os.makedirs('submissions/nn', exist_ok=True)
submission.to_csv('submissions/nn/out.csv', index=False)
plt.show()
