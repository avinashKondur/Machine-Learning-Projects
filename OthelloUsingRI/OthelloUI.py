"""" Othello game GUI

The following is code borrowed from https://github.com/humbhenri/pyOthello/blob/master/ui.py


Original code is written by - Humberto Henrique Campos Pinheiro

We have modified a little bit to meet our project requirements

Avinash Konduru, Nikhila Chireddy

"""

import pygame
import sys
from pygame.locals import QUIT,MOUSEBUTTONDOWN,KEYDOWN
import time
from Constants import BLACK, WHITE, HUMAN, COMPUTER,EASY,HARD
import os


class Gui:

    def __init__(self):

        """ Initializes graphics. """
        pygame.init()

        # colors
        self.BLACK = (0, 0, 0)
        self.BACKGROUND = (0, 0, 255)
        self.WHITE = (255, 255, 255)

        # display
        self.SCREEN_SIZE = (720, 560)
        self.BOARD_POS = (100, 20)
        self.BOARD = (120, 40)
        self.BOARD_SIZE = 400
        self.SQUARE_SIZE = 50
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)

        # messages
        self.BLACK_LAB_POS = (5, self.SCREEN_SIZE[1] / 4)
        self.WHITE_LAB_POS = (560, self.SCREEN_SIZE[1] / 4)
        
        self.HUMAN_LAB_POS = (100, 460) 
        self.COMPUTER_LAB_POS = (350, 460)
        
        self.font = pygame.font.SysFont("Times New Roman", 22)
        self.scoreFont = pygame.font.SysFont("Serif", 58)
        self.scoreFont_Player = pygame.font.SysFont("Serif", 30)
        
        # image files
        self.board_img = pygame.image.load(
            os.path.join("res", "board.bmp")).convert()
        self.black_img = pygame.image.load(
            os.path.join("res", "preta.bmp")).convert()
        self.white_img = pygame.image.load(
            os.path.join("res", "branca.bmp")).convert()
        self.tip_img = pygame.image.load(
            os.path.join("res", "tip.bmp")).convert()
        self.clear_img = pygame.image.load(
            os.path.join("res", "nada.bmp")).convert()

    def show_options(self):

        """ Shows game options screen and returns chosen options
        """
        # default values
        player1 = HUMAN
        player2 = COMPUTER
        
        #level = DEFAULT_LEVEL

        while True:
            self.screen.fill(self.BACKGROUND)
            title_fnt = pygame.font.SysFont("Times New Roman", 34)
            title = title_fnt.render("Othello", True, self.WHITE)
            title_pos = title.get_rect(centerx=self.screen.get_width() / 2,centery=60)

            #start_txt = self.font.render("Start", True, self.WHITE)
            #start_pos = start_txt.get_rect(centerx=self.screen.get_width() / 2,centery=220)
            
            hard_txt = self.font.render(HARD, True, self.WHITE)
            hard_pos = hard_txt.get_rect(centerx=self.screen.get_width() / 2,centery=260)

            easy_txt = self.font.render(EASY, True, self.WHITE)
            easy_pos = easy_txt.get_rect(centerx=self.screen.get_width() / 2,centery=300)

            self.screen.blit(title, title_pos)
            #self.screen.blit(start_txt, start_pos)
            
            self.screen.blit(hard_txt, hard_pos)
            self.screen.blit(easy_txt, easy_pos)


            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == MOUSEBUTTONDOWN:
                    (mouse_x, mouse_y) = pygame.mouse.get_pos()
                    if hard_pos.collidepoint(mouse_x, mouse_y):
                        return (player1, player2, HARD)
                    elif easy_pos.collidepoint(mouse_x, mouse_y):
                        return (player1, player2, EASY)

            pygame.display.flip()

            # desafoga a cpu

    def show_winner(self, player_color):
        self.screen.fill(pygame.Color(0, 0, 0, 50))
        font = pygame.font.SysFont("Courier New", 34)
        
        if player_color == WHITE:
            msg = font.render("White player wins", True, self.WHITE)
        elif player_color == BLACK:
            msg = font.render("Black player wins", True, self.WHITE)
        else:
            msg = font.render("Tie !", True, self.WHITE)

        self.screen.blit(msg, msg.get_rect(centerx=self.screen.get_width() / 2, centery=120))

        pygame.display.flip()
        
        time.sleep(.10)

    def show_game(self):

        """ Game screen. """

        # draws initial screen
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill(self.BACKGROUND)
        self.score_size = 50
        self.score1 = pygame.Surface((self.score_size, self.score_size))
        self.score2 = pygame.Surface((self.score_size, self.score_size))
        self.screen.blit(self.background, (0, 0), self.background.get_rect())
        self.screen.blit(self.board_img, self.BOARD_POS, self.board_img.get_rect())

        #information about color playing by the players        
        human = self.scoreFont_Player.render(HUMAN, True, self.BLACK, self.BACKGROUND)
        computer = self.scoreFont_Player.render(COMPUTER, True, self.WHITE, self.BACKGROUND)
        self.screen.blit(human, (self.HUMAN_LAB_POS[0], self.HUMAN_LAB_POS[1] + 40))
        self.screen.blit(computer, (self.COMPUTER_LAB_POS[0], self.COMPUTER_LAB_POS[1] + 40))

        self.put_stone((3, 3), WHITE)
        self.put_stone((4, 4), WHITE)
        self.put_stone((3, 4), BLACK)
        self.put_stone((4, 3), BLACK)

        pygame.display.flip()

    def put_stone(self, pos, color):
        """ draws piece with given position and color """
        if pos == None:
            return



        # flip orientation (because xy screen orientation)
        pos = (pos[1], pos[0])

        if color == BLACK:
            img = self.black_img
        elif color == WHITE:
            img = self.white_img
        else:
            img = self.tip_img

        x = pos[0] * self.SQUARE_SIZE + self.BOARD[0]
        y = pos[1] * self.SQUARE_SIZE + self.BOARD[1]

        self.screen.blit(img, (x, y), img.get_rect())
        pygame.display.flip()

    def clear_square(self, pos):

        """ Puts in the given position a background image, to simulate that the
        piece was removed.
        """

        # flip orientation
        pos = (pos[1], pos[0])

        x = pos[0] * self.SQUARE_SIZE + self.BOARD[0]
        y = pos[1] * self.SQUARE_SIZE + self.BOARD[1]

        self.screen.blit(self.clear_img, (x, y), self.clear_img.get_rect())

        pygame.display.flip()


    def get_mouse_input(self):
        """ Get place clicked by mouse
        """
        while True:
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONDOWN:
                    (mouse_x, mouse_y) = pygame.mouse.get_pos()

                    # click was out of board, ignores
                    if mouse_x > self.BOARD_SIZE + self.BOARD[0] or mouse_x < self.BOARD[0] or mouse_y > self.BOARD_SIZE + self.BOARD[1] or mouse_y < self.BOARD[1]:
                        continue


                    # find place
                    position = ( (mouse_x - self.BOARD[0]) / self.SQUARE_SIZE), ((mouse_y - self.BOARD[1]) / self.SQUARE_SIZE)

                    # flip orientation
                    position = (position[1], position[0])

                    return position


                elif event.type == QUIT:
                    sys.exit(0)

            time.sleep(.05)


    def update(self, board, blacks, whites):
        """Updates screen

        """
        for i in range(8):
            for j in range(8):
                if board[i][j] != 0:
                    self.put_stone((i, j), board[i][j])

        blacks_str = '%02d ' % int(blacks)
        whites_str = '%02d ' % int(whites)

        self.showScore(blacks_str, whites_str)

        pygame.display.flip()

    def showScore(self, blackStr, whiteStr):

        text = self.scoreFont.render(blackStr, True, self.BLACK, self.BACKGROUND)

        text2 = self.scoreFont.render(whiteStr, True, self.WHITE, self.BACKGROUND)

        self.screen.blit(text, (self.BLACK_LAB_POS[0], self.BLACK_LAB_POS[1] + 40))

        self.screen.blit(text2, (self.WHITE_LAB_POS[0], self.WHITE_LAB_POS[1] + 40))


    def wait_quit(self):
        # wait user to close window
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                break

if __name__ == 'main':
    ui = Gui()
    ui.show_game()