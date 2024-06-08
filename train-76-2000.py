import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train():
    df = pd.read_csv("random_tic_tac_toe_games_dataset.csv", index_col=0)
    print(df.shape)

    # Преобразование категориальных признаков в числовые
    label_encoders = {}
    for col in df.columns[:-1]:  # Все столбцы, кроме 'best_step'
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # Сохраняем LabelEncoder для каждого столбца
    print("label_encoders", label_encoders)

    # Преобразование признаков в one-hot encoding
    categorical_features = df.columns[:-1].tolist()
    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')

    # Определим признаки (X) и целевую переменную (y)
    X = df.drop(columns=['best_step'])
    y = df['best_step']

    # Разделим данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение ColumnTransformer на обучающих данных
    ct.fit(X_train)

    # Преобразование обучающих и тестовых данных
    X_train_transformed = ct.transform(X_train)
    X_test_transformed = ct.transform(X_test)

    # Определение моделей
    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)

    # Комбинированная модель VotingClassifier
    ensemble_model = VotingClassifier(estimators=[
        ('rf', rf_model),
        ('gb', gb_model)
    ], voting='soft')

    # Создание пайплайна с one-hot encoding и моделью
    pipeline = Pipeline([
        ('classifier', ensemble_model)
    ])

    # Определение параметров для GridSearchCV с использованием правильных имен параметров
    param_grid = {
        'classifier__rf__n_estimators': [50, 100, 200],
        'classifier__rf__max_depth': [3, 4, 5],
        'classifier__gb__n_estimators': [50, 100, 200],
        'classifier__gb__learning_rate': [0.01, 0.1, 0.2],
        'classifier__gb__max_depth': [3, 4, 5]
    }

    # Использование GridSearchCV для поиска лучших гиперпараметров
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_transformed, y_train)

    # Лучшая модель
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    # Кросс-валидация для оценки модели
    cv_scores = cross_val_score(best_model, X_train_transformed, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}")

    # Предсказания на тестовом наборе
    y_pred = best_model.predict(X_test_transformed)

    # Оценка точности модели
    accuracy = accuracy_score(y_test, y_pred)
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

    new_board_df = pd.DataFrame([new_board_encoded], columns=categorical_features)
    new_board_transformed = ct.transform(new_board_df)

    # Предсказание лучшего следующего хода
    best_step_prediction = best_model.predict(new_board_transformed)
    print(f"Predicted best step: {best_step_prediction}")
