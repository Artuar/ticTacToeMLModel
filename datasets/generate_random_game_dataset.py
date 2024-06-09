import pandas as pd
import random

from minimax.tic_tac_toe import human, computer, default_board, is_game_finished, get_tic_tac_toe_best_step_list


def generate_random_board():
    while True:
        # create base board
        board = default_board[:]

        # define count of human and computer steps
        num_h = random.randint(0, 4)  # random count for human
        num_c_options = [num_h, num_h - 1]  # possible count for computer before next step
        num_c_options = [x for x in num_c_options if 0 <= x <= 4]  # filter wrong steps
        num_c = random.choice(num_c_options)

        # fill game board with player steps
        positions = random.sample(range(9), num_h + num_c)
        for i in range(num_h):
            board[positions[i]] = human
        for i in range(num_h, num_h + num_c):
            board[positions[i]] = computer

        # check we have at least 1 free cell for next step
        if is_game_finished(board):
            continue

        return board


def generate_unique_boards(num_boards):
    unique_boards = set()

    while len(unique_boards) < num_boards:
        board = tuple(generate_random_board())
        unique_boards.add(board)

    unique_boards = [list(board) for board in unique_boards]

    # create DataFrame from uniq boards
    columns = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9']
    df = pd.DataFrame(unique_boards, columns=columns)

    return df


def generate_random_game_dataset(count=1000):
    df = generate_unique_boards(count)

    best_steps = []
    for _, row in df.iterrows():
        board = row.tolist()
        best_step_result = get_tic_tac_toe_best_step_list(board)
        best_step = best_step_result['bestPositions']
        best_steps.append(best_step)

    df['best_step'] = best_steps

    df.to_csv("datasets/optimal_steps_tic_tac_toe_games_dataset.csv")

    return df
