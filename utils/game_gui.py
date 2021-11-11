import pygame
from sys import exit
from random import choice
import matplotlib.pyplot as plt
from utils.game import Game
from utils.player import *
from utils.constants import *


class GameGUI:
    """
    A class that represents the GUI of the game
    """
    def __init__(self):
        """
        Initialize the GUI
        """
        # Initialize the game
        self.game = Game()
        # Initialize the GUI screen
        pygame.init()
        self.screen = pygame.display.set_mode(GUI_SIZE)
        pygame.display.update()
        # Define font to display main headers
        self.font = pygame.font.SysFont('monospace', 75)
        # Define font to display sub-headings
        self.options_font = pygame.font.SysFont('monospace', 50)

    def draw_main_menu(self):
        """
        Draw main menu with different playing modes of the game
        :return: nothing
        """
        pygame.draw.rect(self.screen, BLACK, (0, 0, GUI_SIZE[0], GUI_SIZE[1]))
        title = self.font.render(GAME_NAME, True, WHITE)
        font_size = self.font.size(GAME_NAME)
        self.screen.blit(title, ((GUI_SIZE[0] - font_size[0]) // 2, 150))
        # Display multi-player option on main menu
        multi_player = self.options_font.render(GAME_MODES[0], True, WHITE)
        font_size = self.options_font.size(GAME_MODES[0])
        corner = (GUI_SIZE[0] - font_size[0]) // 2, 300
        self.screen.blit(multi_player, corner)
        self.rect_multi_player = multi_player.get_rect(topleft=corner)
        # Display play with computer option on main menu
        play_computer = self.options_font.render(GAME_MODES[1], True, WHITE)
        font_size = self.options_font.size(GAME_MODES[1])
        corner = (GUI_SIZE[0] - font_size[0]) // 2, 400
        self.screen.blit(play_computer, corner)
        self.rect_play_computer = play_computer.get_rect(topleft=corner)
        # Update the GUI screen
        pygame.display.update()

    def draw_board(self):
        """
        Draw the empty configuration of the game
        :return: nothing
        """
        # Draw rectangle to represent the board area where coins go in
        pygame.draw.rect(self.screen, BLACK, (0, 0, GUI_SIZE[0], CELL_SIZE))
        # Draw rectangle to represent the board area where coins go in
        pygame.draw.rect(self.screen, BLUE, (0, CELL_SIZE, GUI_SIZE[0], GUI_SIZE[1] - CELL_SIZE))
        # Draw circles to represent dots in the game
        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                # Get center of the next circle to be drawn on the GUI
                center = (j * CELL_SIZE + (CELL_SIZE // 2)), ((i + 1) * CELL_SIZE + (CELL_SIZE // 2))
                # Draw the circle on the GUI
                pygame.draw.circle(self.screen, BLACK, center, RADIUS)
        # Update GUI
        pygame.display.update()

    def main_menu(self):
        """
        Method to implement the logic behind the menu
        :return: mode of the game the user wants to play
        """
        main_menu = True
        play_game = -1
        self.draw_main_menu()
        while main_menu:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    position = pygame.mouse.get_pos()
                    if self.rect_multi_player.collidepoint(position):
                        main_menu = False
                        play_game = 0
                    elif self.rect_play_computer.collidepoint(position):
                        main_menu = False
                        play_game = 1
                if event.type == pygame.QUIT:
                    exit()
        return play_game

    def run_game(self, game_mode, train=False, train_mode=0):
        """
        Function to check for events occurring inside the GUI screen
        Executes entire functioning of the game
        :param game_mode: mode of the game, i.e., multi-player or single-player
        :param train: set true is you want to train the robot
        :param train_mode: type of agent to train with
        :return: nothing
        """
        if game_mode == 0:
            self.run_multi_player()
        else:
            if train:
                self.train_with_agent(trainer=train_mode)
            else:
                self.run_single_player()

    def run_single_player(self, game_status=False):
        q_player = QPlayer(epsilon=0.01)
        player = choice((HUMAN_PLAYER, Q_ROBOT))
        reward = REWARD_NOTHING
        current_board = self.game.current_state
        possible_moves = self.game.get_valid_locations(current_board)
        q_player.train(possible_moves, reward, player, self.game)
        while not game_status:
            if not player:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                    elif event.type == pygame.MOUSEMOTION:
                        pygame.draw.rect(self.screen, BLACK, (0, 0, GUI_SIZE[1], CELL_SIZE))
                        col_index = int(event.pos[0] / CELL_SIZE)
                        center = ((col_index * CELL_SIZE) + (CELL_SIZE // 2)), (CELL_SIZE // 2)
                        pygame.draw.circle(self.screen, RED, center, RADIUS)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        pygame.draw.rect(self.screen, BLACK, (0, 0, GUI_SIZE[1], CELL_SIZE))
                        col_index = int(event.pos[0] / CELL_SIZE)
                        row_index = self.game.get_open_row(col_index, self.game.current_state)
                        if row_index != -1:
                            center = (((col_index * CELL_SIZE) + (CELL_SIZE // 2)),
                                      (((BOARD_SIZE[0] - row_index) * CELL_SIZE) + (CELL_SIZE // 2)))
                            self.game.add_player_token(row_index, col_index, player)
                            self.game.update_current_state(self.game.board)
                            pygame.draw.circle(self.screen, RED, center, RADIUS)
                            if self.game.is_winning_move(row_index, col_index, player):
                                label = self.font.render('Player ' + str(player + 1) + ' Wins!', True, RED)
                                self.screen.blit(label, (40, 10))
                                reward = REWARD_LOSS
                                game_status = True
                            if self.game.is_draw():
                                self.screen.blit('GAME HAS TIED', True, WHITE)
                                reward = REWARD_DRAW
                                game_status = True
                            # Update turn of player
                            player = (player + 1) % 2
            else:
                move = q_player.get_optimal_move(current_board, possible_moves)
                pygame.draw.rect(self.screen, BLACK, (0, 0, GUI_SIZE[1], CELL_SIZE))
                center = ((move[1] * CELL_SIZE) + (CELL_SIZE // 2)), (CELL_SIZE // 2)
                pygame.draw.circle(self.screen, YELLOW, center, RADIUS)
                # Update GUI and wait for some time
                pygame.display.update()
                pygame.time.wait(500)
                # Get center of the token location on the board and draw a circle there
                center = (((move[1] * CELL_SIZE) + (CELL_SIZE // 2)),
                          (((BOARD_SIZE[0] - move[0]) * CELL_SIZE) + (CELL_SIZE // 2)))
                pygame.draw.circle(self.screen, YELLOW, center, RADIUS)
                self.game.add_player_token(move[0], move[1], player)
                self.game.update_current_state(self.game.board)
                if self.game.is_winning_move(move[0], move[1], player):
                    label = self.font.render('Computer Wins!', True, YELLOW)
                    self.screen.blit(label, (40, 10))
                    reward = REWARD_WIN
                    game_status = True
                if self.game.is_draw():
                    reward = REWARD_DRAW
                    game_status = True
                player = (player + 1) % 2
            # Update GUI
            pygame.display.update()
            # Train the computer player
            current_board = self.game.current_state
            possible_moves = self.game.get_valid_locations(current_board)
            q_player.train(possible_moves, reward, player, self.game)
            if game_status:
                q_player.save_memory()
                pygame.time.wait(5000)

    def run_multi_player(self, game_status=False):
        """
        Method to run multi-player (2 players') mode
        :param game_status: set false to start the game
        :return: nothing
        """
        player = choice((HUMAN_PLAYER, HUMAN_PLAYER + 1))
        while not game_status:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(self.screen, BLACK, (0, 0, GUI_SIZE[1], CELL_SIZE))
                    col_index = int(event.pos[0] / CELL_SIZE)
                    center = ((col_index * CELL_SIZE) + (CELL_SIZE // 2)), (CELL_SIZE // 2)
                    if player:
                        pygame.draw.circle(self.screen, YELLOW, center, RADIUS)
                    else:
                        pygame.draw.circle(self.screen, RED, center, RADIUS)
                elif event.type == pygame.MOUSEBUTTONUP:
                    pygame.draw.rect(self.screen, BLACK, (0, 0, GUI_SIZE[1], CELL_SIZE))
                    col_index = int(event.pos[0] / CELL_SIZE)
                    row_index = self.game.get_open_row(col_index, self.game.current_state)
                    if row_index != -1:
                        center = (((col_index * CELL_SIZE) + (CELL_SIZE // 2)),
                                  (((BOARD_SIZE[0] - row_index) * CELL_SIZE) + (CELL_SIZE // 2)))
                        self.game.add_player_token(row_index, col_index, player)
                        if player:
                            pygame.draw.circle(self.screen, YELLOW, center, RADIUS)
                        else:
                            pygame.draw.circle(self.screen, RED, center, RADIUS)
                        if self.game.is_winning_move(row_index, col_index, player):
                            if player:
                                label = self.font.render('Player ' + str(player + 1) + ' Wins!', True, YELLOW)
                            else:
                                label = self.font.render('Player ' + str(player + 1) + ' Wins!', True, RED)
                            self.screen.blit(label, (40, 10))
                            game_status = True
                        if self.game.is_draw():
                            self.screen.blit('GAME HAS TIED', True, WHITE)
                            game_status = True
                        # Update turn of player
                        player = (player + 1) % 2
                # Update GUI
                pygame.display.update()
                if game_status:
                    pygame.time.wait(3000)

    def train_with_agent(self, trainer=0):
        """
        Method to train the game robot using Q-learning
        :param trainer: type of agent to train the game robot
        :return: nothing
        """
        player_win_data, agent_win_data, player_moves, agent_moves = [0], [0], [0], [0]
        draw = [0]
        x = list(range(0, (100 * ITERATIONS) + 1, 100))
        learning_player = QPlayer(epsilon=1.0)
        if trainer == 1:
            trained_player = QPlayer(token=0, mem_location='memory/memory1.npy')
        elif trainer == 2:
            trained_player = MiniMaxPlayer(token=0)
        else:
            trained_player = RandomPlayer(token=0)
        for i in range(ITERATIONS):
            print(i)
            learning_player.save_memory()
            if trainer == 1:
                trained_player.save_memory('memory/memory1.npy')
            player_moves[-1] = player_moves[-1] / ITERATIONS
            agent_moves[-1] = agent_moves[-1] / ITERATIONS

            player_win_data.append(0)
            agent_win_data.append(0)
            player_moves.append(0)
            agent_moves.append(0)
            draw.append(0)
            for j in range(ITERATIONS):
                prev_move = None, None
                player = choice((Q_ROBOT, RANDOM_ROBOT))
                reward = REWARD_NOTHING
                reward1 = REWARD_NOTHING
                # Initialize the game
                self.game = Game()
                game_status = False
                current_board = self.game.current_state
                possible_moves = self.game.get_valid_locations(current_board)
                learning_player.train(possible_moves, reward, learning_player.token, self.game)
                if trainer == 1:
                    trained_player.train(possible_moves, reward1, trained_player.token, self.game)
                elif trainer == 2:
                    trained_player.depth = MINI_MAX_DEPTH
                while not game_status:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            exit()
                    if player == Q_ROBOT:
                        move = learning_player.get_optimal_move(current_board, possible_moves)
                        self.game.add_player_token(move[0], move[1], learning_player.token)
                        player_moves[-1] += 1
                    else:
                        if trainer == 1:
                            move = trained_player.get_optimal_move(current_board, possible_moves)
                        elif trainer == 2:
                            trained_player.depth = MINI_MAX_DEPTH
                            move = trained_player.mini_max(deepcopy(self.game), prev_move,
                                                           (learning_player.token, trained_player.token),
                                                           -float('inf'), float('inf'), True)[0]
                        else:
                            move = trained_player.make_move(possible_moves)
                        self.game.add_player_token(move[0], move[1], trained_player.token)
                        agent_moves[-1] += 1
                    self.game.update_current_state(self.game.board)
                    if self.game.is_winning_move(move[0], move[1], learning_player.token):
                        game_status = True
                        reward, reward1 = REWARD_WIN, REWARD_LOSS
                        player_win_data[-1] += 1
                    elif self.game.is_winning_move(move[0], move[1], trained_player.token):
                        game_status = True
                        reward, reward1 = REWARD_LOSS, REWARD_WIN
                        agent_win_data[-1] += 1
                    elif self.game.is_draw():
                        game_status = True
                        reward, reward1 = REWARD_DRAW, REWARD_DRAW
                        draw[-1] += 1
                    # Update player turn
                    player = 3 - player
                    # Update previous mode
                    prev_move = move
                    # Train the Q-learning players
                    current_board = self.game.current_state
                    possible_moves = self.game.get_valid_locations(current_board)
                    learning_player.train(possible_moves, reward, learning_player.token, self.game)
                    if trainer == 1:
                        trained_player.train(possible_moves, reward1, trained_player.token, self.game)
                    if game_status:
                        if learning_player.exploration_chance > learning_player.epsilon_min:
                            learning_player.exploration_chance *= learning_player.epsilon_decay
        # Take average of moves of the last iterations
        player_moves[-1] = player_moves[-1] / ITERATIONS
        agent_moves[-1] = agent_moves[-1] / ITERATIONS
        # PLot the graphs
        plt.figure(1)
        plt.title('Winning rate over games')
        plt.xlabel('No. of games')
        plt.ylabel('Average winning rate')
        plt.plot(x, player_win_data, color='orange', label='Q-Learning Player')
        plt.plot(x, agent_win_data, color='blue', label='Trained Agent')
        plt.legend()
        plt.show()
        plt.figure(2)
        plt.title('Moves over games')
        plt.xlabel('No. of games')
        plt.ylabel('Average moves')
        plt.plot(x, player_moves, color='orange', label='Q-Learning Player')
        plt.plot(x, agent_moves, color='blue', label='Trained Agent')
        plt.legend()
        plt.show()
