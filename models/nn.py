import os
from time import asctime, localtime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
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

float_features = ["Age", "RoomService",
                  "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "most_spent", "least_spent", "std_spent", "total_spent"]

label_encoders = ["FirstName",
                  "LastName",
                  "num", "GroupId", ]
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


# define nn model parameters
LAYER_SIZES = [512, 256, 128, 1]
ACTIVATION_FUNCTION = "swish"
DROPOUT_SIZE = 0.5
BATCH_SIZE = 256
EPOCHS = 500
y_preds = []
val_scores = []


def run_model(x_t, y_t, x_v, y_v, test):
    """
    @params x_t, y_t, x_v, y_v are the training data and validation data respectively
    @return history is the dataframe containg the loss and validation loss for a run
    Function used to train the model and predict the test data
    """
    # convert to float32 for tf
    X_train = np.asarray(x_t).astype('float32')
    X_test = np.asarray(x_v).astype('float32')
    Y_train = np.asarray(y_t).astype('float32')
    Y_test = np.asarray(y_v).astype('float32')

    # nn model
    input_shape = X_train.shape[1]
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=[input_shape]),
        layers.Dense(input_shape=[input_shape], units=LAYER_SIZES[0], activation=ACTIVATION_FUNCTION,
                     ),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_SIZE),
        layers.Dense(units=LAYER_SIZES[1], activation=ACTIVATION_FUNCTION,
                     ),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_SIZE),
        layers.Dense(units=LAYER_SIZES[2], activation=ACTIVATION_FUNCTION,
                     ),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_SIZE),
        layers.Dense(units=LAYER_SIZES[len(
            LAYER_SIZES)-1], activation="sigmoid"),
    ])

    # compile model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )

    # set up early_stopping to avoid overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        patience=15,
        min_delta=0.001,
        restore_best_weights=True,
    )
    # fit the model
    history = model.fit(
        x=X_train, y=Y_train,
        validation_data=(X_test, Y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    # create a dataframe of the history
    history_df = pd.DataFrame(history.history)

    # predict the test data, add results to a list to be averaged later on
    x_test = np.asarray(test).astype('float32')
    pred = model.predict(x_test)
    y_preds.append(pred)

    val_scores.append(history_df['val_binary_accuracy'].max())

    return history_df


def plot_results(history):
    """
    @params history df from keras containing the loss and validation loss
    Plots the runs loss versus validation loss to analyze overfitting
    """
    # plot the trainiing loss versus validation loss to confirm no under/overfitting
    print(history.columns)
    history.loc[:, ['loss', 'val_loss', ]].plot()
    print(f"Minimum validation loss: {history['val_loss'].min()}")
    print(f"Best val accuracy: {history['val_binary_accuracy'].max()}")

    # write results to file for evaluating model
    with open("run_results.txt", "a") as file:
        file.write(f"\nTime: {asctime(localtime())}\n")
        file.write(
            f"Val binary accuracy best: {str(history['val_binary_accuracy'].max())}\n")
        file.write(
            f"With layer sizes: {LAYER_SIZES},\nbatch_size: {BATCH_SIZE},\nepochs: {EPOCHS},\nactivation function: {ACTIVATION_FUNCTION},\ndropout size: {DROPOUT_SIZE}\n")


# best score might be fron no groupid

skfold = StratifiedKFold(n_splits=5)
for fold, (train_id, test_id) in enumerate(skfold.split(x_train, y_train)):
    print(f"Fold: {fold}")

    # split into the folds
    X_train = x_train.iloc[train_id]
    Y_train = y_train.iloc[train_id]
    X_test = x_train.iloc[test_id]
    Y_test = y_train.iloc[test_id]

    # run the model on the fold
    history = run_model(x_t=X_train, y_t=Y_train,
                        x_v=X_test, y_v=Y_test, test=x_test)
    print(f"Score from fold {fold}: {val_scores[fold]}")

pred = sum(y_preds) / len(y_preds)
submission['Transported'] = pred
submission['Transported'] = np.where(
    submission['Transported'] > 0.5, True, False)

os.makedirs('submissions/nn', exist_ok=True)
submission.to_csv('submissions/nn/out.csv', index=False)
plt.show()
