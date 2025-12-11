from enum import IntEnum

import numpy as np


class Field(IntEnum):
    EMPTY = 0
    MAX = 1
    MIN = -1


def main():
    board_state = np.zeros((3, 3), dtype=int)
    is_max_turn = True

    mode = choose_mode()
    display_board(board_state)

    while not is_winning_state(board_state) and len(possible_moves(board_state)) > 0:
        if mode == "human_vs_ai" and is_max_turn:
            move = get_player_move(board_state)
            board_state[move] = Field.MAX
        else:
            move = get_best_move(board_state, is_max_turn)
            board_state[move] = Field.MAX if is_max_turn else Field.MIN

        is_max_turn = not is_max_turn
        display_board(board_state)

    display_winner(get_winner(board_state))


def minimax(state, is_max_turn, depth=5, alpha=-np.inf, beta=np.inf):
    if is_winning_state(state) or depth == 0:
        return evaluate_state(state)

    if is_max_turn:
        max_value = -np.inf

        for move in possible_moves(state):
            new_state = make_move(move, state, is_max_turn)

            value = minimax(new_state, False, depth - 1, alpha, beta)
            max_value = max(max_value, value)

            alpha = max(alpha, value)
            if beta <= alpha:
                break

        return max_value

    else:
        min_value = np.inf

        for move in possible_moves(state):
            new_state = make_move(move, state, is_max_turn)

            value = minimax(new_state, True, depth - 1, alpha, beta)
            min_value = min(min_value, value)

            beta = min(beta, value)
            if beta <= alpha:
                break

        return min_value


def heuristic(state):
    max_possible = 0
    min_possible = 0
    lines = []

    for row in state:
        lines.append(row)
    for col in state.T:
        lines.append(col)
    lines.append(np.diag(state))
    lines.append(np.diag(np.fliplr(state)))

    for line in lines:
        has_max = np.any(line == Field.MAX)
        has_min = np.any(line == Field.MIN)
        if has_max and not has_min:
            max_possible += 1
        if has_min and not has_max:
            min_possible += 1

    return max_possible - min_possible


def evaluate_state(state):
    if is_winning_state(state):
        winner = get_winner(state)
        if winner == Field.MAX:
            return np.inf
        elif winner == Field.MIN:
            return -np.inf

    return heuristic(state)


def get_best_move(state, is_max_turn):
    best_move = None
    best_value = -np.inf if is_max_turn else np.inf

    for move in possible_moves(state):
        new_state = make_move(move, state, is_max_turn)
        value = minimax(new_state, not is_max_turn)

        if is_max_turn:
            if value >= best_value:
                best_value = value
                best_move = move
        else:
            if value <= best_value:
                best_value = value
                best_move = move

    return best_move


def possible_moves(state):
    moves = []
    for i in range(3):
        for j in range(3):
            if state[i, j] == Field.EMPTY:
                moves.append((i, j))
    return moves


def make_move(move, state, is_max_turn):
    new_state = state.copy()
    new_state[move] = Field.MAX if is_max_turn else Field.MIN
    return new_state


def is_winning_state(state):
    for row in state:
        if abs(np.sum(row)) == 3:
            return True
    for col in state.T:
        if abs(np.sum(col)) == 3:
            return True
    if abs(np.trace(state)) == 3 or abs(np.trace(np.fliplr(state))) == 3:
        return True
    return False


def get_winner(state):
    for row in state:
        if np.sum(row) == 3:
            return Field.MAX
        elif np.sum(row) == -3:
            return Field.MIN

    for col in state.T:
        if np.sum(col) == 3:
            return Field.MAX
        elif np.sum(col) == -3:
            return Field.MIN

    if np.trace(state) == 3 or np.trace(np.fliplr(state)) == 3:
        return Field.MAX
    elif np.trace(state) == -3 or np.trace(np.fliplr(state)) == -3:
        return Field.MIN

    return None


def get_player_move(state):
    while True:
        move_input = input("insert your move [row col]: ")
        row, col = map(int, move_input.split())
        if 0 <= row < 3 and 0 <= col < 3:
            if state[row, col] == Field.EMPTY:
                return row, col
            else:
                print("cell is already occupied")
        else:
            print("move is out of range")


def choose_mode():
    print("choose game mode:")
    print("1. human vs AI")
    print("2. AI vs AI")

    while True:
        choice = input("enter your choice: ").strip()
        if choice == "1":
            return "human_vs_ai"
        elif choice == "2":
            return "ai_vs_ai"
        else:
            print("invalid input")


def display_board(state):
    print("\n  0   1   2")
    for i, row in enumerate(state):
        print(f"{i} ", end="")
        for j, cell in enumerate(row):
            if cell == Field.EMPTY:
                symbol = " "
            elif cell == Field.MAX:
                symbol = "X"
            else:
                symbol = "O"
            print(f"{symbol}", end="")
            if j < 2:
                print(" | ", end="")
        print()
        if i < 2:
            print("  -----------")
    print()


def display_winner(winner):
    if winner == Field.MAX:
        print("X wins")
    elif winner == Field.MIN:
        print("O wins")
    else:
        print("its a draw")


if __name__ == "__main__":
    main()
