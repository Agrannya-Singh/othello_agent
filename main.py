import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from flask import Flask, request, jsonify
import copy
import random

# Constants for the Othello board
BLACK = 1
WHITE = -1
EMPTY = 0
BOARD_SIZE = 8
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

app = Flask(__name__)
game = None  # Global game instance

# Zobrist hashing setup
ZOBRIST_TABLE = np.random.randint(0, 2**64, size=(BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint64)
TRANSPOSITION_TABLE = {}  # Hash -> (score, depth, flag)

# Difficulty to depth mapping
DIFFICULTY_DEPTHS = {
    'easy': 2,
    'medium': 4,
    'hard': 6
}

class Othello:
    def __init__(self, difficulty='medium'):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.current_player = BLACK
        self.model = self.create_model()
        self.game_states = []  # Store board states and moves for training
        self.game_outcome = None
        self.depth = DIFFICULTY_DEPTHS.get(difficulty.lower(), DIFFICULTY_DEPTHS['medium'])

    def create_model(self):
        """Create a neural network for board evaluation."""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(BOARD_SIZE * BOARD_SIZE,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, 'relu'),
            layers.Dense(1, 'tanh')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def collect_game_data(self):
        """Collect the current board state for training."""
        board_vector = self.board.flatten() / 2.0
        self.game_states.append((board_vector, self.current_player))

    def update_model(self, game_states, game_outcome):
        """Train the neural network using data from a game."""
        if not game_states or game_outcome is None:
            return
        X_train = []
        y_train = []
        for board_vector, player in game_states:
            X_train.append(board_vector)
            if game_outcome == BLACK:
                score = 1.0 if player == BLACK else -1.0
            elif game_outcome == WHITE:
                score = -1.0 if player == BLACK else 1.0
            else:
                score = 0.0
            y_train.append(score)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    def get_heuristic_score(self):
        """Traditional heuristic: piece count difference."""
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        return (black_count - white_count) / (black_count + white_count + 1e-6)

    def compute_hash(self):
        """Compute Zobrist hash for the current board state."""
        hash_value = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board[row][col]
                if piece != EMPTY:
                    index = 0 if piece == BLACK else 1
                    hash_value ^= ZOBRIST_TABLE[row][col][index]
        return hash_value

    def get_valid_moves(self):
        """Return a list of valid moves for the current player."""
        valid_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.is_valid_move(row, col):
                    valid_moves.append((row, col))
        return valid_moves

    def is_valid_move(self, row, col):
        """Check if a move is valid."""
        if self.board[row][col] != EMPTY:
            return False
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == -self.current_player:
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == -self.current_player:
                    r += dr
                    c += dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == self.current_player:
                    return True
        return False

    def make_move(self, move):
        """Make a move and flip pieces."""
        row, col = move
        self.board[row][col] = self.current_player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            to_flip = []
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == -self.current_player:
                to_flip.append((r, c))
                r += dr
                c += dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == self.current_player:
                for fr, fc in to_flip:
                    self.board[fr][fc] = self.current_player
        self.collect_game_data()

    def is_game_over(self):
        """Check if the game is over."""
        return not self.get_valid_moves() and not Othello(self.board, -self.current_player).get_valid_moves()

    def get_winner(self):
        """Determine the winner based on piece count."""
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        if black_count > white_count:
            return BLACK
        elif white_count > black_count:
            return WHITE
        return 0

    def evaluate_board(self, use_ml=True):
        """Evaluate the board using the neural network or heuristic."""
        if use_ml and self.model is not None:
            board_vector = self.board.flatten() / 2.0
            score = self.model.predict(np.array([board_vector]), verbose=0)[0][0]
            return score * self.current_player
        return self.get_heuristic_score()

    def minimax(self, depth, alpha, beta, maximizing_player):
        """Minimax with Alpha-Beta pruning and Zobrist hashing."""
        hash_value = self.compute_hash()
        if hash_value in TRANSPOSITION_TABLE:
            score, stored_depth, flag = TRANSPOSITION_TABLE[hash_value]
            if stored_depth >= depth:
                if flag == 'exact':
                    return score
                elif flag == 'lowerbound' and score > alpha:
                    alpha = score
                elif flag == 'upperbound' and score < beta:
                    beta = score
                if alpha >= beta:
                    return score

        if depth == 0 or self.is_game_over():
            score = self.evaluate_board()
            TRANSPOSITION_TABLE[hash_value] = (score, depth, 'exact')
            return score

        valid_moves = self.get_valid_moves()
        if not valid_moves:
            score = self.evaluate_board()
            TRANSPOSITION_TABLE[hash_value] = (score, depth, 'exact')
            return score

        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                new_game = Othello(difficulty=self.depth)
                new_game.board = self.board.copy()
                new_game.current_player = self.current_player
                new_game.make_move(move)
                eval = new_game.minimax(depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            flag = 'exact' if max_eval <= alpha else 'lowerbound'
            TRANSPOSITION_TABLE[hash_value] = (max_eval, depth, flag)
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_game = Othello(difficulty=self.depth)
                new_game.board = self.board.copy()
                new_game.current_player = self.current_player
                new_game.make_move(move)
                eval = new_game.minimax(depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            flag = 'exact' if min_eval >= beta else 'upperbound'
            TRANSPOSITION_TABLE[hash_value] = (min_eval, depth, flag)
            return min_eval

    def get_best_move(self):
        """Get the best move using Minimax with Alpha-Beta pruning and Zobrist hashing."""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return None
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for move in valid_moves:
            new_game = Othello(difficulty=self.depth)
            new_game.board = self.board.copy()
            new_game.current_player = self.current_player
            new_game.make_move(move)
            score = new_game.minimax(self.depth - 1, alpha, beta, False)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_move

    def self_play_and_train(self, num_games=10):
        """Simulate AI vs. AI games and train the model."""
        global TRANSPOSITION_TABLE
        for _ in range(num_games):
            TRANSPOSITION_TABLE.clear()  # Clear transposition table for each game
            game_states = []
            current_game = Othello(difficulty=self.depth)
            while not current_game.is_game_over():
                valid_moves = current_game.get_valid_moves()
                if not valid_moves:
                    current_game.current_player *= -1
                    continue
                move = current_game.get_best_move()
                current_game.make_move(move)
                game_states.extend(current_game.game_states)
                current_game.game_states = []  # Clear for next move
            outcome = current_game.get_winner()
            self.update_model(game_states, outcome)
        return {"message": f"Trained on {num_games} self-play games"}

@app.route('/start_game', methods=['POST'])
def start_game():
    global game
    data = request.json
    difficulty = data.get('difficulty', 'medium')
    game = Othello(difficulty=difficulty)
    return jsonify({
        'status': 'success',
        'board': game.board.flatten().tolist(),
        'current_player': game.current_player,
        'game_states': len(game.game_states),
        'depth': game.depth
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    global game
    if not game:
        return jsonify({'status': 'error', 'message': 'Game not started'}), 400
    data = request.json
    move = (data['row'], data['col'])
    if move not in game.get_valid_moves():
        return jsonify({'status': 'error', 'message': 'Invalid move'}), 400
    game.make_move(move)
    ai_move = game.get_best_move() if game.current_player == WHITE else None
    if ai_move:
        game.make_move(ai_move)
    game.current_player *= -1
    game_over = game.is_game_over()
    if game_over:
        game.game_outcome = game.get_winner()
        game.update_model(game.game_states, game.game_outcome)
    return jsonify({
        'status': 'success',
        'board': game.board.flatten().tolist(),
        'current_player': game.current_player,
        'ai_move': ai_move,
        'game_over': game_over,
        'winner': game.game_outcome if game_over else None
    })

@app.route('/suggest_move', methods=['GET'])
def suggest_move():
    global game
    if not game:
        return jsonify({'status': 'error', 'message': 'Game not started'}), 400
    move = game.get_best_move()
    return jsonify({'status': 'success', 'move': move})

@app.route('/train', methods=['POST'])
def train():
    global game
    if not game or not game.game_states or game.game_outcome is None:
        return jsonify({'status': 'error', 'message': 'No game data to train'}), 400
    game.update_model(game.game_states, game.game_outcome)
    return jsonify({'status': 'success', 'message': 'Model trained'})

@app.route('/board_state', methods=['GET'])
def board_state():
    global game
    if not game:
        return jsonify({'status': 'error', 'message': 'Game not started'}), 400
    return jsonify({
        'status': 'success',
        'board': game.board.flatten().tolist(),
        'current_player': game.current_player
    })

@app.route('/self_train', methods=['POST'])
def self_train():
    global game
    if not game:
        return jsonify({'status': 'error', 'message': 'Game not started'}), 400
    data = request.json
    num_games = data.get('num_games', 10)
    result = game.self_play_and_train(num_games)
    return jsonify({'status': 'success', **result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
