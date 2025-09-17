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
        self.screen = pygame.display.set_mode((640, 640))
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
        self.font = pygame.font.Font(None, 36)

    def run(self):
        while True:
            self.input()
            if not self.game_over:
                self.update()
            self.draw()

    def input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

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
        self.grid.insertToken(self.grid.gridLogic, self.currentPlayer, row, col)
        tiles = self.grid.swappableTiles(row, col, self.grid.gridLogic, self.currentPlayer)
        for tile in tiles:
            self.grid.gridLogic[tile[0]][tile[1]] *= -1
        
        self.consecutive_passes = 0  # Reset pass counter
        self.switchPlayer()

    def switchPlayer(self):
        self.currentPlayer *= -1
        
        # Check if current player has valid moves
        validMoves = self.grid.findAvailMoves(self.grid.gridLogic, self.currentPlayer)
        if not validMoves:
            self.consecutive_passes += 1
            if self.consecutive_passes >= 2:
                self.game_over = True
                return
            else:
                # Current player must pass, switch back
                self.currentPlayer *= -1

    def update(self):
        # AI move (only in AI mode and when it's AI's turn)
        if (self.game_mode == "ai" and self.currentPlayer == self.player2 and not self.game_over):
            pygame.time.wait(500)
            validMoves = self.grid.findAvailMoves(self.grid.gridLogic, self.currentPlayer)
            if validMoves:
                move = self.minimaxDecision()
                y, x = move
                self.makeMove(y, x)

    def minimaxDecision(self):
        depth = 3
        validMoves = self.grid.findAvailMoves(self.grid.gridLogic, self.currentPlayer)
        bestScore = -math.inf
        bestMove = None
        for move in validMoves:
            y, x = move
            gridCopy = copy.deepcopy(self.grid.gridLogic)
            self.grid.insertToken(gridCopy, self.currentPlayer, y, x)
            tiles = self.grid.swappableTiles(y, x, gridCopy, self.currentPlayer)
            for tile in tiles:
                gridCopy[tile[0]][tile[1]] *= -1
            score = self.minValue(gridCopy, depth-1)
            if score > bestScore:
                bestScore = score
                bestMove = move
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
        self.screen.fill((0, 128, 0))  # Green background
        self.grid.drawGrid(self.screen)
        
        # Display current player and score
        white_score, black_score = self.getScore()
        
        if not self.game_over:
            current_text = "White's turn" if self.currentPlayer == 1 else "Black's turn"
            if self.game_mode == "ai" and self.currentPlayer == self.player2:
                current_text = "AI thinking..."
        else:
            if white_score > black_score:
                current_text = "White wins!"
            elif black_score > white_score:
                current_text = "Black wins!"
            else:
                current_text = "It's a tie!"

        # Display text
        text_surface = self.font.render(current_text, True, (255, 255, 255))
        score_surface = self.font.render(f"White: {white_score} | Black: {black_score}", True, (255, 255, 255))
        
        self.screen.blit(text_surface, (10, 650))
        self.screen.blit(score_surface, (400, 650))
        
        pygame.display.update()

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
                rect = pygame.Rect(col * 80, row * 80, 80, 80)
                pygame.draw.rect(window, (0, 100, 0), rect)
                pygame.draw.rect(window, (0, 0, 0), rect, 1)
                cell = self.gridLogic[row][col]
                if cell != 0:
                    colour = self.whitetoken_colour if cell == 1 else self.blacktoken_colour
                    pygame.draw.circle(window, colour, rect.center, 30)

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
