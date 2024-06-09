# Tic-Tac-Toe with Machine Learning

This project provides the opportunity to play Tic-Tac-Toe against a computer using a trained machine learning model to predict the best moves.

# Requirements

Python 3.6+

# Install Dependencies

Install the required dependencies using pip:

```sh
pip install flask joblib scikit-learn xgboost
```

# Generate the Dataset

To generate the dataset for training the model, call the `generate_random_game_dataset()` function from the `datasets/generate_random_game_dataset.py` file:

```sh
python -c "from datasets.generate_random_game_dataset import generate_random_game_dataset; generate_random_game_dataset(2000)"
```

# Train the Model

After generating the dataset, train the model by calling the train() function from the model/train_gradient.py file:

```sh
python -c "from model.train_gradient import train; train()"
```

# Play the Game Against the Computer

After training the model, you can try playing against the computer by running the web application:

```sh
python app.py
```

Open your browser and go to http://127.0.0.1:5000/ to start the game.

# Project Structure
- minimax/: Contains minimax algorithm for searching optimal game move.
- datasets/: Contains scripts for generating the dataset.
- model/: Contains scripts for training the model.
- app.py: Flask application for playing Tic-Tac-Toe against the computer.
- index.html: HTML file for displaying the game board.

# License

This project is licensed under the MIT License. 