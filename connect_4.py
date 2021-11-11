from utils.game_gui import GameGUI


if __name__ == '__main__':
    # Initialize GUI screen and draw the game board on it
    gui = GameGUI()
    game_mode = gui.main_menu()
    gui.draw_board()
    gui.run_game(game_mode, train=False, train_mode=2)
