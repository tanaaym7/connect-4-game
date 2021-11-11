import math
import random
import pickle
from numpy import ravel_multi_index
from copy import deepcopy
from utils.constants import *


class Player:
    """
    A class to represent various types of players in the game
    By default, it represents a human player
    """
    def __init__(self, token=HUMAN_PLAYER):
        """
        Initialize player with its token
        :param token: integer that represents the player on the board
        """
        self.token = token

    def make_move(self):
        """
        Method to make a move on the board
        :return: nothing
        """
        pass


class QPlayer(Player):
    """
    A class that represents a Q-learning based AI player
    """
    def __init__(self, token=Q_ROBOT, alpha=0.9, gamma=0.75, epsilon=0.9, mem_location=MEM_LOCATION):
        """
        Initialize the player with its token, learning rate, discount factor, and exploration chance
        :param token: integer that represents the player on the board
        :param alpha: learning rate of the AI player; value between 0-1
        :param gamma: discount factor for future rewards; value between 0-1
        :param epsilon: probability of random exploration; value between 0-1
        """
        Player.__init__(self, token)
        self.discount_factor = gamma
        self.learning_rate = alpha
        self.exploration_chance = epsilon
        self.q_value = {}
        self.load_memory(mem_location)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def load_memory(self, file_location=MEM_LOCATION):
        with open(file_location, 'rb') as memory:
            self.q_value = pickle.load(memory)
        memory.close()

    def save_memory(self, file_location=MEM_LOCATION):
        with open(file_location, 'wb') as memory:
            pickle.dump(self.q_value, memory)
        memory.close()

    @staticmethod
    def get_key(state):
        return str(tuple(map(tuple, state)))

    def get_optimal_move(self, current_state, moves):
        # Allow the robot to explore random moves
        if random.random() < self.exploration_chance:
            return random.choice(moves)
        # Get maximum Q-value
        max_q_value, q_values = self.get_max_q_value(current_state, moves)
        # Check if there are more than 1 instances of maximum Q-value
        if q_values.count(max_q_value) > 1:
            # Choose a random instance among the various instances of maximum Q-value
            max_options = [i for i in range(len(moves)) if q_values[i] == max_q_value]
            selected_option = random.choice(max_options)
            return moves[selected_option]
        # Return move with the max Q-value
        return moves[q_values.index(max_q_value)]

    def get_q_value(self, state, move):
        """
        Method to retrieve Q-value based on the state-action pair
        :param state: State of the game whose Q-value needs to be extracted
        :param move: Action done on the state
        :return: Q-value of the given state-action pair
        """
        # Convert move into an index
        move = str(ravel_multi_index(move, dims=BOARD_SIZE))
        # Convert state-action pair into a string
        key = self.get_key(state)
        # Check if the state-action pair exists
        if self.q_value.get(key) is None:
            self.q_value[key] = {move: 1.0}
        else:
            if self.q_value[key].get(move) is None:
                self.q_value[key][move] = 1.0
        return self.q_value[key][move]

    def calc_q_value(self, reward, prev_q_value, max_q_value):
        """
        Method to calculate Q-value
        :param reward: Reward based on the current state of the board
        :param prev_q_value: Q-value of the previous state of the board
        :param max_q_value: Maximum Q-value out of the current state of the board and available moves
        :return: Q-value of the given state
        """
        # Apply the Bellman's equation
        return prev_q_value + self.learning_rate * (reward + (self.discount_factor * max_q_value) - prev_q_value)

    def get_max_q_value(self, state, moves):
        """
        Method to get the maximum Q-value of the current state of the board
        :param state: Current or next state of the game
        :param moves: a list of possible moves on the board
        :return: maximum Q-value of the current state of the board and list of all Q-values
        """
        # Generate a list of Q-values for each state-action pair
        q_values = [self.get_q_value(state, move) for move in moves]
        return max(q_values), q_values

    def train(self, valid_moves, reward, player, game):
        """
        Method to train the robot of the game using Q-Learning
        :param valid_moves: a list of possible moves on the board
        :param reward: reward based on the current status of the board
        :param player: ID of the player to train
        :param game: an instance of the Game class
        :return: nothing
        """
        for move in valid_moves:
            current_state = deepcopy(game.current_state)
            # Get the Q-value of the previous state of the game
            prev_q_value = self.get_q_value(current_state, move)
            next_state = deepcopy(game.current_state)
            next_state[move[0]][move[1]] = player + 1
            next_moves = game.get_valid_locations(next_state)
            if len(next_moves):
                # Get the maximum Q-value of the current state of the game
                max_q_value, _ = self.get_max_q_value(next_state, next_moves)
                # Update Q-value of the state-action pair
                move = str(ravel_multi_index(move, dims=BOARD_SIZE))
                self.q_value[self.get_key(current_state)][move] = self.calc_q_value(reward, prev_q_value, max_q_value)


class RandomPlayer(Player):
    """
    A class that represents a player that picks moves randomly
    """

    def __init__(self, token=RANDOM_ROBOT):
        """
        Initialize the human player with its token
        :param token: integer that represents the player on the board
        """
        Player.__init__(self, token)

    def make_move(self, valid_moves):
        """
        Method to make a move on the board
        :return: a tuple containing location of the token to be placed
        """
        return random.choice(valid_moves)


class MiniMaxPlayer(Player):
    """
    A class that represents a player trained using the MinMax algorithm
    """
    def __init__(self, token=MINI_MAX_ROBOT):
        """
        Initialize the min-max player with its token
        :param token: integer that represents the player on the board
        """
        Player.__init__(self, token)
        self.window_length = 4
        self.win_score = 100000000000
        self.depth = MINI_MAX_DEPTH

    @staticmethod
    def evaluate_window(window, token_player, token_opponent):
        # Initialize current window score as 0
        score = 0
        # Count the no. of minmax player's tokens in the window
        player_token_count = window.count(token_player)
        # Appropriate reward according to the status of the window
        if player_token_count == 4:
            score += 100
        elif player_token_count == 3 and window.count(NO_TOKEN) == 1:
            score += 5
        elif player_token_count == 2 and window.count(NO_TOKEN) == 2:
            score += 2
        elif window.count(token_opponent) == 3 and window.count(NO_TOKEN) == 1:
            score -= 4
        # Return the final score for the given window
        return score

    def score_position(self, board, token, token_opp):
        # Initialize score for the current status of the board
        score = 0
        # Score center column
        center_array = [int(i) for i in list(board[:, BOARD_SIZE[1]//2])]
        center_count = center_array.count(token)
        score += center_count * 3
        # Score Horizontal
        for r in range(BOARD_SIZE[0]):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(BOARD_SIZE[1]-3):
                window = row_array[c:c+self.window_length]
                score += self.evaluate_window(window, token, token_opp)
        # Score Vertical
        for c in range(BOARD_SIZE[1]):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(BOARD_SIZE[0]-3):
                window = col_array[r:r+self.window_length]
                score += self.evaluate_window(window, token, token_opp)
        # Score positive sloped diagonal
        for r in range(BOARD_SIZE[0]-3):
            for c in range(BOARD_SIZE[1]-3):
                window = [board[r+i][c+i] for i in range(self.window_length)]
                score += self.evaluate_window(window, token, token_opp)
        # Score negative sloped diagonal
        for r in range(BOARD_SIZE[0]-3):
            for c in range(BOARD_SIZE[1]-3):
                window = [board[r+3-i][c+i] for i in range(self.window_length)]
                score += self.evaluate_window(window, token, token_opp)
        # Return final score for the current board
        return score

    def mini_max(self, game, last_token_pos, player_tokens, alpha, beta, maximizing_player):
        if last_token_pos != (None, None):
            # Evaluate scores for various game terminating scenarios
            if game.is_winning_move(last_token_pos[0], last_token_pos[1], player_tokens[1]):
                return None, self.win_score
            elif game.is_winning_move(last_token_pos[0], last_token_pos[1], player_tokens[0]):
                return None, -self.win_score
            elif game.is_draw():  # Game is over, no more valid moves
                return None, 0
        # Depth is 0
        if self.depth == 0:
            return None, self.score_position(game.board, player_tokens[1], player_tokens[0])
        # Start minimax algorithm
        # Get valid move locations from the board
        valid_locations = game.get_valid_locations(game.board)

        if maximizing_player:
            value = -math.inf
            move = random.choice(valid_locations)
            for loc in valid_locations:
                game_copy = deepcopy(game)
                game_copy.add_player_token(loc[0], loc[1], player_tokens[1])
                self.depth -= 1
                new_score = self.mini_max(game_copy, loc, player_tokens, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    move = loc
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return move, value
        else:
            value = math.inf
            move = random.choice(valid_locations)
            for loc in valid_locations:
                game_copy = deepcopy(game)
                game_copy.add_player_token(loc[0], loc[1], player_tokens[1])
                self.depth -= 1
                new_score = self.mini_max(game_copy, loc, player_tokens, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    move = loc
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return move, value
