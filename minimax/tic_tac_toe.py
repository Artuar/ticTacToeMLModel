import random
from typing import List, Union, Dict

# players
computer = "c"
human = "h"

# board types
Point = Union[int, str]
Board = List[Point]

# board
default_board: Board = [
    0, 1, 2,
    3, 4, 5,
    6, 7, 8
]


# check if player won
def is_win(board: Board, player: computer or human) -> bool:
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    for combination in winning_combinations:
        if all(board[i] == player for i in combination):
            return True
    return False


# check if board is full
def is_full_board(board: Board) -> bool:
    return all(point == computer or point == human for point in board)


def is_game_finished(board: Board):
    return is_full_board(board) or is_win(board, computer) or is_win(board, human)


# Minimax recursive algorithm to calculate value of a step
def calculate_value(position: int, board: Board, player: str, deep: int) -> int:
    expected_board: Board = board[:]
    expected_board[position] = player

    if is_win(expected_board, computer):
        return 10 - deep

    if is_win(expected_board, human):
        return deep - 10

    if is_full_board(expected_board):
        return 0

    next_player = human if player == computer else computer
    next_step = calculate_position_values(expected_board, next_player, deep + 1)
    entries = list(next_step.values())

    if next_player == human:
        return min(entries)
    else:
        return max(entries)


# check every possible steps
def calculate_position_values(board: Board, player: str, deep: int = 0) -> Dict[int, int]:
    result = {}
    for position, point in enumerate(board):
        if point == computer or point == human:
            continue
        result[position] = calculate_value(position, board, player, deep)
    return result


# choose step with higher scope
def choose_best_position(best_positions: Dict[int, int]) -> List[int]:
    max_value = max(best_positions.values())
    best_positions_list = [position for position, value in best_positions.items() if value == max_value]
    return best_positions_list


# get next best step
def get_tic_tac_toe_best_step(board: Board, random_best_position: bool = False) -> dict:
    best_positions = calculate_position_values(board, computer)
    best_positions = choose_best_position(best_positions)

    if random_best_position:
        return {"board": board, "bestPositions": random.choice(best_positions)}
    else:
        return {"board": board, "bestPositions": best_positions[0]}

# get next best steps
def get_tic_tac_toe_best_step_list(board: Board) -> dict:
    best_positions = calculate_position_values(board, computer)
    best_positions = choose_best_position(best_positions)

    return {"board": board, "bestPositions": best_positions}
