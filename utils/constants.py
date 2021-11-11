# Row-by-column representation of the board
BOARD_SIZE = 6, 7
# Define size of each cell in the GUI of the game
CELL_SIZE = 100
# Define radius of dot
RADIUS = (CELL_SIZE // 2) - 5
# Define size of GUI screen
GUI_SIZE = (BOARD_SIZE[0] + 1) * CELL_SIZE, (BOARD_SIZE[1]) * CELL_SIZE
# Define various colors used on the GUI of the game
RED = 255, 0, 0
BLUE = 0, 0, 255
BLACK = 0, 0, 0
WHITE = 255, 255, 255
YELLOW = 255, 255, 0
# Define various players in the Game
HUMAN_PLAYER = 0
Q_ROBOT = 1
RANDOM_ROBOT = 2
MINI_MAX_ROBOT = 3
# Define various rewards
REWARD_WIN = 1
REWARD_LOSS = -1
REWARD_DRAW = 0.2
REWARD_NOTHING = 0
# Define various modes of the game
GAME_NAME = 'CONNECT-4'
GAME_MODES = {0: '2 Players', 1: 'vs Computer'}
# Define various modes of learning
LEARNING_MODES = {'random_agent': 0, 'trained_agent': 1, 'minimax_agent': 2}
NO_TOKEN = 0
# Define location of memory file
MEM_LOCATION = 'memory/memory.npy'
# Define variable to store no. of iterations to train the game robot
ITERATIONS = 100
# Define depth of mini-max player
MINI_MAX_DEPTH = 5
