import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import ast

def train():
    df = pd.read_csv("random_tic_tac_toe_games_dataset.csv", index_col=0)
    print(df.shape)

    # Преобразование категориальных признаков в числовые
    label_encoders = {}
    for col in df.columns[:-1]:  # Все столбцы, кроме 'best_step'
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # Сохраняем LabelEncoder для каждого столбца

    # Преобразование признаков в one-hot encoding
    categorical_features = df.columns[:-1].tolist()
    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')

    # Преобразуем массив возможных лучших ходов в числовой формат
    df['best_step'] = df['best_step'].apply(ast.literal_eval)

    # Определим признаки (X) и целевую переменную (y)
    X = df.drop(columns=['best_step'])
    y = df['best_step']

    # Используем MultiLabelBinarizer для преобразования целевых переменных в бинарные массивы
    mlb = MultiLabelBinarizer()
    y_mlb = mlb.fit_transform(y)

    # Разделим данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y_mlb, test_size=0.2, random_state=42)

    # Обучение ColumnTransformer на обучающих данных
    ct.fit(X_train)

    # Преобразование обучающих и тестовых данных
    X_train_transformed = ct.transform(X_train)
    X_test_transformed = ct.transform(X_test)

    # Определение моделей
    rf_model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
    gb_model = OneVsRestClassifier(GradientBoostingClassifier(random_state=42))

    # Создание пайплайна с one-hot encoding и моделью
    rf_pipeline = Pipeline([
        ('classifier', rf_model)
    ])

    gb_pipeline = Pipeline([
        ('classifier', gb_model)
    ])

    # Определение параметров для GridSearchCV с использованием правильных имен параметров
    rf_param_grid = {
        'classifier__estimator__n_estimators': [50, 100, 200],
        'classifier__estimator__max_depth': [3, 4, 5]
    }

    gb_param_grid = {
        'classifier__estimator__n_estimators': [50, 100, 200],
        'classifier__estimator__learning_rate': [0.01, 0.1, 0.2],
        'classifier__estimator__max_depth': [3, 4, 5]
    }

    # Определение метрики для оценки точности
    def custom_scorer(y_true, y_pred):
        y_true = mlb.inverse_transform(y_true)
        y_pred = mlb.inverse_transform(y_pred)
        return np.mean([1 if set(pred) & set(true) else 0 for pred, true in zip(y_pred, y_true)])

    scorer = make_scorer(custom_scorer, greater_is_better=True)

    # Использование GridSearchCV для поиска лучших гиперпараметров
    rf_grid_search = GridSearchCV(estimator=rf_pipeline, param_grid=rf_param_grid, cv=5, n_jobs=-1, scoring=scorer)
    gb_grid_search = GridSearchCV(estimator=gb_pipeline, param_grid=gb_param_grid, cv=5, n_jobs=-1, scoring=scorer)

    rf_grid_search.fit(X_train_transformed, y_train)
    gb_grid_search.fit(X_train_transformed, y_train)

    # Лучшая модель
    rf_best_model = rf_grid_search.best_estimator_
    gb_best_model = gb_grid_search.best_estimator_

    print(f"Best parameters found for RandomForest: {rf_grid_search.best_params_}")
    print(f"Best parameters found for GradientBoosting: {gb_grid_search.best_params_}")

    # Кросс-валидация для оценки модели
    rf_cv_scores = cross_val_score(rf_best_model, X_train_transformed, y_train, cv=5, scoring=scorer)
    gb_cv_scores = cross_val_score(gb_best_model, X_train_transformed, y_train, cv=5, scoring=scorer)

    print(f"RandomForest Cross-validation scores: {rf_cv_scores}")
    print(f"RandomForest Mean CV accuracy: {rf_cv_scores.mean():.2f}")

    print(f"GradientBoosting Cross-validation scores: {gb_cv_scores}")
    print(f"GradientBoosting Mean CV accuracy: {gb_cv_scores.mean():.2f}")

    # Предсказания на тестовом наборе
    rf_y_pred = rf_best_model.predict(X_test_transformed)
    gb_y_pred = gb_best_model.predict(X_test_transformed)

    # Оценка точности модели
    rf_accuracy = custom_scorer(y_test, rf_y_pred)
    gb_accuracy = custom_scorer(y_test, gb_y_pred)

    print(f"RandomForest Accuracy: {rf_accuracy:.2f}")
    print(f"GradientBoosting Accuracy: {gb_accuracy:.2f}")

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
    rf_best_step_prediction = rf_best_model.predict(new_board_transformed)
    gb_best_step_prediction = gb_best_model.predict(new_board_transformed)

    print(f"RandomForest Predicted best step: {mlb.inverse_transform(rf_best_step_prediction)}")
    print(f"GradientBoosting Predicted best step: {mlb.inverse_transform(gb_best_step_prediction)}")

train()