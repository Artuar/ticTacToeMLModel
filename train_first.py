import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    df = pd.read_csv("random_step_tic_tac_toe_games_dataset.csv", index_col=0)
    print(df.shape)

    # convert string data to integer
    label_encoders = {}
    for col in df.columns[:-1]:  # all columns apart from 'best_step'
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # save LabelEncoder for every column

    # define the features (X) and the target variable (y)
    X = df.drop(columns=['best_step'])
    y = df['best_step']

    # separate data to learning and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create and learn model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # predict in test sample
    y_pred = model.predict(X_test)

    # value model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
