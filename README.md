# Tic-Tac-Toe

Tic-Tac-Toe implementation featuring an unbeatable AI based on the **Minimax algorithm with Alpha-Beta Pruning**.

## Algorithm

The AI explores all possible game states to select the best move. Alpha-Beta Pruning optimization reduces the number of evaluated nodes, making the search significantly faster.

## Heuristic Function

The heuristic evaluates non-terminal game states based on:

1. **Positional weights**:
    - Center: 4 points (controls 4 lines)
    - Corners: 3 points (control 3 lines)
    - Edges: 2 points (control 2 lines)

2. **Line evaluation** (rows, columns, diagonals):
    - 2 symbols in a row (unblocked): +/-5 points
    - 1 symbol in a row (unblocked): +/-1 point
    - Blocked lines: 0 points

Terminal states are evaluated as `+inf` (MAX wins), `-inf` (MIN wins), or heuristic score (ongoing game).

## Installation

```bash
git clone https://github.com/nonofare/tic_tac_toe
cd tic_tac_toe
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

Choose game mode:

- **1. Human vs AI** - Play as X against AI (O)
- **2. AI vs AI** - Watch two AIs play against each other

Enter moves as `row col` (e.g., `1 1` for center).