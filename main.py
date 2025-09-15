import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from flask import Flask, request, jsonify, session
import copy
import random

# Constants for the Othello board
BLACK = 1
WHITE = -1
EMPTY = 0
BOARD_SIZE = 8
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

app = Flask(__name__)
app.secret_key = '^21Nk£=2f{#1#3w^d-.+[7%a$Z/VXt!Z7_dSg*%7Y£8JK4Fr0O'  #this will be roatwd into the environment svariable and be chnaged this is NOT AN API key rather an emcription key for the coookies 

# Global model for shared learning across all games/sessions
#intially set as zero because why not 
GLOBAL_MODEL = None

def create_global_model():
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        GLOBAL_MODEL = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(BOARD_SIZE * BOARD_SIZE,)),#relu for hidden layers of the neuron 
            #tanh for outer / surface neuron layers
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='tanh')
        ])
        GLOBAL_MODEL.compile(optimizer='adam', loss='mse')
    return GLOBAL_MODEL

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
        self.model = create_global_model()  # Shared model for evaluation and training
        self.game_states = []  # Store board states and players for training
        self.game_outcome = None
        self.depth = DIFFICULTY_DEPTHS.get(difficulty.lower(), DIFFICULTY_DEPTHS['medium'])

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
        if self.get_valid_moves():
            return False
        temp_game = copy.deepcopy(self)
        temp_game.current_player *= -1
        return not temp_game.get_valid_moves()

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
        """Minimax with Alpha-Beta pruning (no hashing or transposition table)."""
        if depth == 0 or self.is_game_over():
            return self.evaluate_board()

        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return self.evaluate_board()

        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                new_game = copy.deepcopy(self)
                new_game.make_move(move)
                new_game.current_player *= -1
                eval = new_game.minimax(depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_game = copy.deepcopy(self)
                new_game.make_move(move)
                new_game.current_player *= -1
                eval = new_game.minimax(depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self):
        """Get the best move using Minimax with Alpha-Beta pruning."""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return None
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for move in valid_moves:
            new_game = copy.deepcopy(self)
            new_game.make_move(move)
            new_game.current_player *= -1
            score = new_game.minimax(self.depth - 1, alpha, beta, False)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_move

    def self_play_and_train(self, num_games=10):
        """Simulate AI vs. AI games and train the model."""
        for _ in range(num_games):
            game_states = []
            current_game = Othello(difficulty=self.depth)
            while not current_game.is_game_over():
                valid_moves = current_game.get_valid_moves()
                if not valid_moves:
                    current_game.current_player *= -1
                    continue
                move = current_game.get_best_move()
                if move is None:
                    break
                current_game.make_move(move)
                current_game.current_player *= -1
                game_states.extend(current_game.game_states)
                current_game.game_states = []  # Clear for next move
            outcome = current_game.get_winner()
            self.update_model(game_states, outcome)
        return {"message": f"Trained on {num_games} self-play games"}

def serialize_game(game):
    """Serialize Othello instance to dict for session storage."""
    return {
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'game_states': [(state.tolist(), player) for state, player in game.game_states],
        'game_outcome': game.game_outcome,
        'depth': game.depth
    }

def deserialize_game(data):
    """Deserialize dict to Othello instance."""
    game = Othello()  # Difficulty not needed since depth is set later
    game.board = np.array(data['board'])
    game.current_player = data['current_player']
    game.game_states = [(np.array(state), player) for state, player in data['game_states']]
    game.game_outcome = data['game_outcome']
    game.depth = data['depth']
    return game

@app.route('/start_game', methods=['POST'])
def start_game():
    data = request.json
    difficulty = data.get('difficulty', 'medium')
    game = Othello(difficulty=difficulty)
    session['game'] = serialize_game(game)
    return jsonify({
        'status': 'success',
        'board': game.board.flatten().tolist(),
        'current_player': game.current_player,
        'game_states': len(game.game_states),
        'depth': game.depth
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    if 'game' not in session:
        return jsonify({'status': 'error', 'message': 'Game not started'}), 400
    game = deserialize_game(session['game'])
    data = request.json
    try:
        row = int(data['row'])
        col = int(data['col'])
    except (KeyError, ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid row or col'}), 400
    move = (row, col)
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return jsonify({'status': 'error', 'message': 'Move out of bounds'}), 400
    if move not in game.get_valid_moves():
        return jupytext({'status': 'error', 'message': 'Invalid move'}), 400
    game.make_move(move)
    game.current_player *= -1  # Switch to AI
    ai_move = None
    if game.current_player == WHITE:  # Assuming AI plays as WHITE
        ai_move = game.get_best_move()
        if ai_move:
            game.make_move(ai_move)
            game.current_player *= -1  # Switch back to player
    game_over = game.is_game_over()
    if game_over:
        game.game_outcome = game.get_winner()
        game.update_model(game.game_states, game.game_outcome)
    session['game'] = serialize_game(game)
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
    if 'game' not in session:
        return jsonify({'status': 'error', 'message': 'Game not started'}), 400
    game = deserialize_game(session['game'])
    move = game.get_best_move()
    return jsonify({'status': 'success', 'move': move})

@app.route('/train', methods=['POST'])
def train():
    if 'game' not in session:
        return jsonify({'status': 'error', 'message': 'Game not started'}), 400
    game = deserialize_game(session['game'])
    if not game.game_states or game.game_outcome is None:
        return jsonify({'status': 'error', 'message': 'No game data to train'}), 400
    game.update_model(game.game_states, game.game_outcome)
    session['game'] = serialize_game(game)
    return jsonify({'status': 'success', 'message': 'Model trained'})

@app.route('/board_state', methods=['GET'])
def board_state():
    if 'game' not in session:
        return jsonify({'status': 'error', 'message': 'Game not started'}), 400
    game = deserialize_game(session['game'])
    return jsonify({
        'status': 'success',
        'board': game.board.flatten().tolist(),
        'current_player': game.current_player
    })

@app.route('/self_train', methods=['POST'])
def self_train():
    if 'game' not in session:
        return jsonify({'status': 'error', 'message': 'Game not started'}), 400
    game = deserialize_game(session['game'])
    data = request.json
    num_games = data.get('num_games', 10)
    result = game.self_play_and_train(num_games)
    session['game'] = serialize_game(game)
    return jsonify({'status': 'success', **result})


@app.route('/')
def index():
    # This multi-line string contains the full HTML and CSS for the webpage. 
    # do not worry about this part this is just a simple gretting page to for our api page where we will do machine learning on the othello's res[ponses 
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OthelloAI Dojo</title>
        <style>
            body {
                background-color: #121212;
                color: #e0e0e0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                margin: 0;
                padding: 2rem;
                text-align: center;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            h1 {
                font-size: 3rem;
                color: #ffffff;
                margin-bottom: 0.5rem;
            }
            h3 {
                font-size: 1.2rem;
                color: #bb86fc;
                font-weight: 400;
                margin-top: 0;
            }
            p {
                font-size: 1.1rem;
                line-height: 1.6;
            }
            .project-by {
                margin: 2rem 0;
                font-style: italic;
                color: #a0a0a0;
            }
            .image-placeholder {
                background-color: #333;
                border: 2px dashed #555;
                padding: 4rem 1rem;
                margin: 2rem 0;
                border-radius: 8px;
                color: #888;
            }
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                text-align: left;
                margin-top: 3rem;
            }
            .feature-item {
                background-color: #1e1e1e;
                padding: 1.5rem;
                border-radius: 8px;
                border: 1px solid #333;
            }
            .feature-item h4 {
                margin-top: 0;
                color: #bb86fc;
                font-size: 1.25rem;
            }
            .cta-button {
                display: inline-block;
                background-color: #03dac6;
                color: #121212;
                padding: 1rem 2rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                font-size: 1.2rem;
                margin-top: 2rem;
                transition: transform 0.2s ease;
            }
            .cta-button:hover {
                transform: scale(1.05);
            }
            hr {
                border: none;
                border-top: 1px solid #333;
                margin: 3rem 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OthelloAI Dojo 🧠</h1>
            <h3>Challenge a Modern AI in the Classic Game of Strategy</h3>
            <p class="project-by">A Project by the <strong>IEEE Information Theory Society, VIT Vellore Chapter</strong>.</p>
            
            <p>Welcome to the <strong>OthelloAI Dojo</strong>! Step into a modern arena where the timeless game of Othello (also known as Reversi) meets cutting-edge artificial intelligence. Built with Next.js, this interactive web app lets you test your skills against a powerful AI opponent. It's more than just a game; it's a chance to learn, strategize, and witness AI decision-making firsthand.</p>
            
            <div class="image-placeholder"></div>

            <hr>

            <h2>Features at a Glance</h2>

            <div class="features-grid">
                <div class="feature-item">
                    <h4>Interactive & Intuitive Gameplay</h4>
                    <p>Engage in a classic 8x8 Othello match on a clean, responsive, and fully interactive game board. Our UI highlights all your valid moves, so you can focus on strategy.</p>
                </div>
                <div class="feature-item">
                    <h4>A Formidable AI Opponent</h4>
                    <p>Ready for a challenge? Play against an AI with adjustable difficulty levels (Easy, Medium, Hard). Can you outsmart the machine?</p>
                </div>
                <div class="feature-item">
                    <h4>Learn from the Master with Gemini AI</h4>
                    <p>Stuck on a move? Ask for a suggestion and receive not only the optimal move but also the strategic rationale behind it.</p>
                </div>
                <div class="feature-item">
                    <h4>Visualize the AI's Mind</h4>
                    <p>Get a natural language explanation of the AI's thought process for its moves and deepen your understanding of the game.</p>
                </div>
            </div>

            <a href="#" class="cta-button">&gt;&gt; Enter the Dojo &amp; Start a Game &lt;&lt;</a>
        </div>
    </body>
    </html>
    """
    return html_content



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
