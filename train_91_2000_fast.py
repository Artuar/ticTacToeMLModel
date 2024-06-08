import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import ast
from xgboost import XGBClassifier

def train():
    df = pd.read_csv("optimal_steps_tic_tac_toe_games_dataset.csv", index_col=0)
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

    # Определение модели XGBoost
    xgb_model = OneVsRestClassifier(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))

    # Создание пайплайна с one-hot encoding и моделью
    xgb_pipeline = Pipeline([
        ('classifier', xgb_model)
    ])

    # Определение параметров для GridSearchCV
    xgb_param_grid = {
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
    xgb_grid_search = GridSearchCV(estimator=xgb_pipeline, param_grid=xgb_param_grid, cv=5, n_jobs=-1, scoring=scorer)

    xgb_grid_search.fit(X_train_transformed, y_train)

    # Лучшая модель
    xgb_best_model = xgb_grid_search.best_estimator_

    print(f"Best parameters found for XGBoost: {xgb_grid_search.best_params_}")

    # Кросс-валидация для оценки модели
    xgb_cv_scores = cross_val_score(xgb_best_model, X_train_transformed, y_train, cv=5, scoring=scorer)

    print(f"XGBoost Cross-validation scores: {xgb_cv_scores}")
    print(f"XGBoost Mean CV accuracy: {xgb_cv_scores.mean():.2f}")

    # Предсказания на тестовом наборе
    xgb_y_pred = xgb_best_model.predict(X_test_transformed)

    # Оценка точности модели
    xgb_accuracy = custom_scorer(y_test, xgb_y_pred)

    print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")

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
    xgb_best_step_prediction = xgb_best_model.predict(new_board_transformed)

    print(f"XGBoost Predicted best step: {mlb.inverse_transform(xgb_best_step_prediction)}")

train()