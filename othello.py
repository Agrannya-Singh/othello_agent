import pygame
import random
import copy
import math

# Utility functions
def directions(x, y, minX=0, minY=0, maxX=7, maxY=7):
    validdirections = []
    if x != minX: validdirections.append((x-1, y))
    if x != minX and y != minY: validdirections.append((x-1, y-1))
    if x != minX and y != maxY: validdirections.append((x-1, y+1))

    if x != maxX: validdirections.append((x+1, y))
    if x != maxX and y != minY: validdirections.append((x+1, y-1))
    if x != maxX and y != maxY: validdirections.append((x+1, y+1))

    if y != minY: validdirections.append((x, y-1))
    if y != maxY: validdirections.append((x, y+1))

    return validdirections

# Classes
class Othello:
    def __init__(self, game_mode="ai"):  # "ai" for AI vs Human, "human" for Human vs Human
        pygame.init()
        self.screen = pygame.display.set_mode((640, 720))  # Extra height for text
        pygame.display.set_caption('Othello')

        self.player1 = 1  # White
        self.player2 = -1 # Black

        self.currentPlayer = self.player1
        self.game_mode = game_mode  # "ai" or "human"

        self.rows = 8
        self.columns = 8
        self.size = (80, 80)

        self.grid = Grid(self.rows, self.columns, self.size, self)
        self.game_over = False
        self.consecutive_passes = 0

        # Font for displaying messages
        pygame.font.init()  # Make sure font system is initialized
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def run(self):
        clock = pygame.time.Clock()
        while True:
            self.input()
            if not self.game_over:
                self.update()
            self.draw()
            clock.tick(60)  # 60 FPS

    def input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            # Handle restart when game is over
            if self.game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.restart_game()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

            # Handle human player moves
            if (self.game_mode == "human" or 
                (self.game_mode == "ai" and self.currentPlayer == self.player1)):
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.game_over:
                        x, y = pygame.mouse.get_pos()
                        x, y = x // 80, y // 80
                        validMoves = self.grid.findAvailMoves(self.grid.gridLogic, self.currentPlayer)
                        if (y, x) in validMoves:
                            self.makeMove(y, x)

    def makeMove(self, row, col):
        print(f"Making move at ({row}, {col}) for player {self.currentPlayer}")
        self.grid.insertToken(self.grid.gridLogic, self.currentPlayer, row, col)
        tiles = self.grid.swappableTiles(row, col, self.grid.gridLogic, self.currentPlayer)
        for tile in tiles:
            self.grid.gridLogic[tile[0]][tile[1]] *= -1
        
        self.consecutive_passes = 0  # Reset pass counter
        
        # Check for game end before switching players
        if not self.checkGameEnd():
            self.switchPlayer()

    def switchPlayer(self):
        old_player = self.currentPlayer
        self.currentPlayer *= -1
        print(f"Switched from player {old_player} to player {self.currentPlayer}")
        
        # Check if current player has valid moves
        validMoves = self.grid.findAvailMoves(self.grid.gridLogic, self.currentPlayer)
        print(f"Player {self.currentPlayer} has {len(validMoves)} valid moves: {validMoves}")
        
        if not validMoves:
            self.consecutive_passes += 1
            print(f"Player {self.currentPlayer} must pass. Consecutive passes: {self.consecutive_passes}")
            
            if self.consecutive_passes >= 2:
                print("Game over - both players passed consecutively")
                self.game_over = True
                return
            else:
                # Current player must pass, switch back
                print(f"Player {self.currentPlayer} passes, switching back to {old_player}")
                self.currentPlayer = old_player
        else:
            self.consecutive_passes = 0  # Reset pass counter when a player can move

    def restart_game(self):
        """Restart the game with the same mode"""
        self.currentPlayer = self.player1
        self.game_over = False
        self.consecutive_passes = 0
        self.grid = Grid(self.rows, self.columns, self.size, self)

    def update(self):
        # AI move (only in AI mode and when it's AI's turn)
        if (self.game_mode == "ai" and self.currentPlayer == self.player2 and not self.game_over):
            pygame.time.wait(500)
            validMoves = self.grid.findAvailMoves(self.grid.gridLogic, self.currentPlayer)
            
            # Debug: print valid moves for AI
            print(f"AI turn - Valid moves: {validMoves}")
            
            if validMoves:
                # Try minimax first, fallback to random if it fails
                move = self.minimaxDecision()
                if move is None:
                    print("Minimax failed, choosing random move")
                    move = random.choice(validMoves)
                
                y, x = move
                print(f"AI chooses move: {move}")
                self.makeMove(y, x)
            else:
                print("AI has no valid moves, switching player")
                self.switchPlayer()

    def minimaxDecision(self):
        depth = 3
        validMoves = self.grid.findAvailMoves(self.grid.gridLogic, self.currentPlayer)
        
        if not validMoves:
            print("No valid moves in minimaxDecision")
            return None
            
        bestScore = -math.inf
        bestMove = None
        
        print(f"AI evaluating {len(validMoves)} moves: {validMoves}")
        
        for move in validMoves:
            y, x = move
            gridCopy = copy.deepcopy(self.grid.gridLogic)
            self.grid.insertToken(gridCopy, self.currentPlayer, y, x)
            tiles = self.grid.swappableTiles(y, x, gridCopy, self.currentPlayer)
            for tile in tiles:
                gridCopy[tile[0]][tile[1]] *= -1
            score = self.minValue(gridCopy, depth-1)
            print(f"Move {move} scored: {score}")
            if score > bestScore:
                bestScore = score
                bestMove = move
        
        print(f"Best move chosen: {bestMove} with score: {bestScore}")
        return bestMove

    def maxValue(self, grid, depth):
        if depth == 0:
            return self.evaluate(grid)
        validMoves = self.grid.findAvailMoves(grid, self.player2)
        if not validMoves:
            return self.evaluate(grid)
        maxEval = -math.inf
        for move in validMoves:
            y, x = move
            gridCopy = copy.deepcopy(grid)
            self.grid.insertToken(gridCopy, self.player2, y, x)
            tiles = self.grid.swappableTiles(y, x, gridCopy, self.player2)
            for tile in tiles:
                gridCopy[tile[0]][tile[1]] *= -1
            eval = self.minValue(gridCopy, depth - 1)
            maxEval = max(maxEval, eval)
        return maxEval

    def minValue(self, grid, depth):
        if depth == 0:
            return self.evaluate(grid)
        validMoves = self.grid.findAvailMoves(grid, self.player1)
        if not validMoves:
            return self.evaluate(grid)
        minEval = math.inf
        for move in validMoves:
            y, x = move
            gridCopy = copy.deepcopy(grid)
            self.grid.insertToken(gridCopy, self.player1, y, x)
            tiles = self.grid.swappableTiles(y, x, gridCopy, self.player1)
            for tile in tiles:
                gridCopy[tile[0]][tile[1]] *= -1
            eval = self.maxValue(gridCopy, depth - 1)
            minEval = min(minEval, eval)
        return minEval

    def evaluate(self, grid):
        # Simple evaluation: count difference in tokens
        count = 0
        for row in grid:
            for cell in row:
                count += cell
        return count

    def checkGameEnd(self):
        """Check if the game should end and update game_over status"""
        player1_moves = self.grid.findAvailMoves(self.grid.gridLogic, self.player1)
        player2_moves = self.grid.findAvailMoves(self.grid.gridLogic, self.player2)
        
        # If neither player has moves, game is over
        if not player1_moves and not player2_moves:
            print("Game over - no moves available for either player")
            self.game_over = True
            return True
            
        # Check if board is full
        empty_squares = 0
        for row in self.grid.gridLogic:
            for cell in row:
                if cell == 0:
                    empty_squares += 1
        
        if empty_squares == 0:
            print("Game over - board is full")
            self.game_over = True
            return True
            
        return False

    def getScore(self):
        white_count = 0
        black_count = 0
        for row in self.grid.gridLogic:
            for cell in row:
                if cell == 1:
                    white_count += 1
                elif cell == -1:
                    black_count += 1
        return white_count, black_count

    def draw(self):
        # Clear screen with green background
        self.screen.fill((0, 128, 0))
        
        # Draw the game board
        self.grid.drawGrid(self.screen)
        
        # Get current scores
        white_score, black_score = self.getScore()
        
        # Debug: Always check current game state
        print(f"Drawing - Game over: {self.game_over}, Current player: {self.currentPlayer}")
        
        # Determine what text to show
        if not self.game_over:
            if self.currentPlayer == 1:
                current_text = "White's Turn"
            else:
                current_text = "Black's Turn"
            if self.game_mode == "ai" and self.currentPlayer == self.player2:
                current_text = "AI is thinking..."
            instruction_text = "Click to place your piece"
        else:
            # Game over - show results
            if white_score > black_score:
                current_text = f"WHITE WINS! ({white_score}-{black_score})"
            elif black_score > white_score:
                current_text = f"BLACK WINS! ({black_score}-{white_score})"
            else:
                current_text = f"TIE GAME! ({white_score}-{black_score})"
            instruction_text = "Press R to restart | ESC to quit"

        # Render text surfaces
        current_surface = self.font.render(current_text, True, (255, 255, 255))
        score_text = f"White: {white_score} | Black: {black_score}"
        score_surface = self.font.render(score_text, True, (255, 255, 255))
        instruction_surface = self.small_font.render(instruction_text, True, (255, 255, 0))
        
        # Position text on screen
        self.screen.blit(current_surface, (10, 650))
        self.screen.blit(score_surface, (350, 650))
        self.screen.blit(instruction_surface, (10, 680))
        
        # Update the display
        pygame.display.flip()  # Use flip() instead of update() for full screen refresh

class Grid:
    def __init__(self, rows, columns, size, game):
        self.GAME = game
        self.y = rows
        self.x = columns
        self.size = size
        self.whitetoken_colour = (255, 255, 255)
        self.blacktoken_colour = (0, 0, 0)

        self.gridLogic = self.regenGrid(self.y, self.x)

    def regenGrid(self, rows, columns):
        grid = [[0 for _ in range(columns)] for _ in range(rows)]
        self.insertToken(grid, 1, 3, 3)
        self.insertToken(grid, -1, 3, 4)
        self.insertToken(grid, 1, 4, 4)
        self.insertToken(grid, -1, 4, 3)
        return grid

    def drawGrid(self, window):
        for row in range(self.y):
            for col in range(self.x):
                # Calculate rectangle position and size
                rect = pygame.Rect(col * 80, row * 80, 80, 80)
                
                # Draw green background for each cell
                pygame.draw.rect(window, (0, 150, 0), rect)
                
                # Draw black border for each cell
                pygame.draw.rect(window, (0, 0, 0), rect, 2)
                
                # Draw piece if there is one
                cell = self.gridLogic[row][col]
                if cell != 0:
                    center_x = col * 80 + 40  # Center of the cell
                    center_y = row * 80 + 40  # Center of the cell
                    
                    if cell == 1:  # White piece
                        pygame.draw.circle(window, self.whitetoken_colour, (center_x, center_y), 30)
                        pygame.draw.circle(window, (0, 0, 0), (center_x, center_y), 30, 2)  # Black border
                    else:  # Black piece (cell == -1)
                        pygame.draw.circle(window, self.blacktoken_colour, (center_x, center_y), 30)
                        pygame.draw.circle(window, (255, 255, 255), (center_x, center_y), 30, 2)  # White border

    def printGameLogicBoard(self):
        print('  | A | B | C | D | E | F | G | H |')
        for i, row in enumerate(self.gridLogic):
            line = f'{i} |'.ljust(3, " ")
            for item in row:
                line += f"{item}".center(3, " ") + '|'
            print(line)
        print()

    def findValidCells(self, grid, curPlayer):
        validCellToClick = []
        for gridX, row in enumerate(grid):
            for gridY, cell in enumerate(row):
                if cell != 0:
                    continue
                DIRECTIONS = directions(gridX, gridY)
                for direction in DIRECTIONS:
                    dirX, dirY = direction
                    checkedCell = grid[dirX][dirY]
                    if checkedCell == 0 or checkedCell == curPlayer:
                        continue
                    if (gridX, gridY) in validCellToClick:
                        continue
                    validCellToClick.append((gridX, gridY))
        return validCellToClick

    def swappableTiles(self, x, y, grid, player):
        surroundCells = directions(x, y)
        swappableTiles = []
        for checkCell in surroundCells:
            checkX, checkY = checkCell
            difX, difY = checkX - x, checkY - y
            currentLine = []
            RUN = True
            while RUN:
                if grid[checkX][checkY] == player * -1:
                    currentLine.append((checkX, checkY))
                elif grid[checkX][checkY] == player:
                    RUN = False
                    break
                elif grid[checkX][checkY] == 0:
                    currentLine.clear()
                    RUN = False
                checkX += difX
                checkY += difY

                if checkX < 0 or checkX > 7 or checkY < 0 or checkY > 7:
                    currentLine.clear()
                    RUN = False

            if len(currentLine) > 0:
                swappableTiles.extend(currentLine)

        return swappableTiles

    def findAvailMoves(self, grid, currentPlayer):
        validCells = self.findValidCells(grid, currentPlayer)
        playableCells = []
        for cell in validCells:
            x, y = cell
            swapTiles = self.swappableTiles(x, y, grid, currentPlayer)
            if len(swapTiles) > 0:
                playableCells.append(cell)
        return playableCells

    def insertToken(self, grid, curplayer, y, x):
        grid[y][x] = curplayer
        
if __name__ == '__main__':
    # For AI vs Human game
    game = Othello("ai")
    # For Human vs Human game, use:
    # game = Othello("human")
    game.run()
    pygame.quit()
