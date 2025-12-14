from enum import IntEnum

import numpy as np


class Field(IntEnum):
    EMPTY = 0
    MAX = 1
    MIN = -1


def main():
    print(">>> Tic Tac Toe <<<")
    game_mode = choose_game_mode()

    state = np.zeros((3, 3), dtype=int)
    draw_state(state)
    is_max_turn = True

    while (not is_winning_state(state)
           and len(possible_moves(state)) > 0):
        if game_mode == "human_vs_ai" and is_max_turn:
            move = get_player_move(state)
            state[move] = Field.MAX
        else:
            move = get_ai_move(state, is_max_turn)
            state[move] = Field.MAX if is_max_turn else Field.MIN

        is_max_turn = not is_max_turn
        draw_state(state)

    print("game over - ", end="")
    winner = get_winner(state)
    match winner:
        case Field.MAX:
            print("X wins")
        case Field.MIN:
            print("O wins")
        case _:
            print("its a draw")


def minimax(state, is_max_turn, depth=9, alpha=-np.inf, beta=np.inf):
    if (is_winning_state(state)
            or len(possible_moves(state)) == 0
            or depth == 0):
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
    score = 0

    position_weights = np.array([
        [3, 2, 3],
        [2, 4, 2],
        [3, 2, 3]
    ])

    score += np.sum((state == Field.MAX) * position_weights) * 0.1
    score -= np.sum((state == Field.MIN) * position_weights) * 0.1

    lines = []
    for row in state:
        lines.append(row)
    for col in state.T:
        lines.append(col)
    lines.append(np.diag(state))
    lines.append(np.diag(np.fliplr(state)))

    for line in lines:
        max_count = np.sum(line == Field.MAX)
        min_count = np.sum(line == Field.MIN)

        if max_count > 0 and min_count == 0:
            if max_count == 2:
                score += 5
            elif max_count == 1:
                score += 1

        elif min_count > 0 and max_count == 0:
            if min_count == 2:
                score -= 5
            elif min_count == 1:
                score -= 1

    return score


def evaluate_state(state):
    if is_winning_state(state):
        winner = get_winner(state)
        if winner == Field.MAX:
            return np.inf
        elif winner == Field.MIN:
            return -np.inf

    return heuristic(state)


def get_ai_move(state, is_max_turn):
    best_move = None
    best_value = -np.inf if is_max_turn else np.inf

    for move in possible_moves(state):
        new_state = make_move(move, state, is_max_turn)
        state_value = minimax(new_state, not is_max_turn)

        if is_max_turn:
            if state_value > best_value:
                best_value = state_value
                best_move = move
        else:
            if state_value < best_value:
                best_value = state_value
                best_move = move

    return best_move


def possible_moves(state):
    free_fields = np.argwhere(state == Field.EMPTY)
    return [tuple(p) for p in free_fields]


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
    if (abs(np.trace(state)) == 3
            or abs(np.trace(np.fliplr(state))) == 3):
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
        try:
            move = input("insert your move [row col]: ")
            row, col = map(int, move.split())
            if 0 <= row < 3 and 0 <= col < 3:
                if state[row, col] == Field.EMPTY:
                    return row, col
                else:
                    print("cell is already occupied")
            else:
                print("move is out of range")
        except ValueError:
            print("invalid format, use: row col (e.g., 1 2)")


def choose_game_mode():
    print("choose game mode:")
    print("1. human vs AI")
    print("2. AI vs AI")

    while True:
        match input("enter your choice: ").strip():
            case "1":
                return "human_vs_ai"
            case "2":
                return "ai_vs_ai"
            case _:
                print("invalid input")


def draw_state(state):
    print("\n  0   1   2")
    for i, row in enumerate(state):
        print(f"{i} ", end="")
        for j, cell in enumerate(row):
            symbol = " "
            match cell:
                case Field.MAX:
                    symbol = "X"
                case Field.MIN:
                    symbol = "O"
            print(f"{symbol}", end="")
            if j < 2:
                print(" | ", end="")
        print()
        if i < 2:
            print("  -----------")
    print()


if __name__ == "__main__":
    main()
