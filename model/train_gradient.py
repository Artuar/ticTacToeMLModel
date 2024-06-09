import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

def train_and_save_model():
    df = pd.read_csv("./datasets/optimal_steps_tic_tac_toe_games_dataset.csv", index_col=0)
    print(df.shape)

    # convert string data to integer
    label_encoders = {}
    for col in df.columns[:-1]:  # all columns apart from 'best_step'
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # save LabelEncoder for every column

    # convert 'best_step' from string to list
    df['best_step'] = df['best_step'].apply(json.loads)

    # Expand the dataset to include all possible best steps
    expanded_data = []
    original_indices = []
    for idx, row in df.iterrows():
        for step in row['best_step']:
            new_row = row.drop('best_step').to_dict()
            new_row['best_step'] = step
            expanded_data.append(new_row)
            original_indices.append(idx)

    expanded_df = pd.DataFrame(expanded_data)
    expanded_df['original_index'] = original_indices

    # define the features (X) and the target variable (y)
    X = expanded_df.drop(columns=['best_step', 'original_index'])
    y = expanded_df['best_step']
    original_indices = expanded_df['original_index']

    # separate data to learning and test samples
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, original_indices, test_size=0.2, random_state=42)

    # create and learn model
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    # predict in test sample
    y_pred = model.predict(X_test)

    # value model accuracy from the list of possible proper steps
    y_test_original = df.loc[test_indices]['best_step'].tolist()
    accuracy = sum([1 if pred in actual else 0 for pred, actual in zip(y_pred, y_test_original)]) / len(y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the model and label encoders
    joblib.dump(model, 'model/tic_tac_toe_model.pkl')
    joblib.dump(label_encoders, 'model/label_encoders.pkl')
