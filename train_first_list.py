import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    df = pd.read_csv("optimal_steps_tic_tac_toe_games_dataset.csv", index_col=0)
    print(df.shape)

    # Преобразование категориальных признаков в числовые
    label_encoders = {}
    for col in df.columns[:-1]:  # Все столбцы, кроме 'best_step'
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # Сохраняем LabelEncoder для каждого столбца

    # Преобразование 'best_step' из строки в список
    df['best_step'] = df['best_step'].apply(eval)

    # Определим признаки (X) и целевую переменную (y)
    X = df.drop(columns=['best_step'])
    y = df['best_step']

    # Разделим данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создаем и обучаем модель
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.apply(lambda x: x[0]))  # Обучаем на первом элементе списка

    # Предсказания на тестовом наборе
    y_pred = model.predict(X_test)

    # Оценка точности модели с учетом списка возможных шагов
    accuracy = sum([1 if pred in actual else 0 for pred, actual in zip(y_pred, y_test)]) / len(y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Пример нового состояния игрового поля
    new_board = [
        'h', 1, 'c',
        3, 'c', 5,
        6, 7, 'h']  # Замените на реальное состояние

    # Преобразование нового состояния игрового поля
    new_board_encoded = []
    for i, item in enumerate(new_board):
        col_name = f'point_{i + 1}'
        if col_name in label_encoders:
            encoded_item = label_encoders[col_name].transform([str(item)])[0]
            new_board_encoded.append(encoded_item)

    # Предсказание лучшего следующего хода
    best_step_prediction = model.predict([new_board_encoded])
    print(f"Predicted best step: {best_step_prediction}")

# Вызов функции train
train()