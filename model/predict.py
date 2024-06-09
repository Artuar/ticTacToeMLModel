import joblib


def load_model_and_predict(new_board):
    # Load the model and label encoders
    model = joblib.load('model/tic_tac_toe_model.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')

    # Convert new board state to encoded form
    new_board_encoded = []
    for i, item in enumerate(new_board):
        col_name = f'point_{i + 1}'
        if col_name in label_encoders:
            encoded_item = label_encoders[col_name].transform([str(item)])[0]
            new_board_encoded.append(encoded_item)

    # Predict the best next step
    best_step_prediction = model.predict([new_board_encoded])
    return best_step_prediction[0]
