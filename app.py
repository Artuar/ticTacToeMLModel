import os

from flask import Flask, request, jsonify, send_from_directory
import joblib

from model.predict import load_model_and_predict
from minimax.tic_tac_toe import computer, Board, is_win, human, is_full_board

app = Flask(__name__)

# Load model
model = joblib.load('model/tic_tac_toe_model.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

def get_game_state(board: Board):
    if is_win(board, computer):
        return 'Computer won!'
    elif is_win(board, human):
        return'You won!'
    elif is_full_board(board):
        return 'Tie!'
    else:
        return 'game'

# handle main page
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# handle computer move
@app.route('/api/move', methods=['POST'])
def get_computer_move():
    data = request.json
    board = data['board']
    if get_game_state(board) != 'game':
        return jsonify({
            'board': board,
            'state': get_game_state(board)
        })

    best_step_prediction = load_model_and_predict(board)
    move = int(best_step_prediction)
    board[move] = computer
    return jsonify({
        'board': board,
        'move': move,
        'state': get_game_state(board)
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)