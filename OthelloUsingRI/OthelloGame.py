#'''

#The following is code borrowed from https://github.com/humbhenri/pyOthello/blob/master/othello.pyw 

#Original code is written by - Humberto Henrique Campos Pinheiro

#We have modified a little bit to meet our project requirements

#Avinash Konduru, Nikhila Chireddy

#'''

import pygame
from OthelloUI import Gui
from Players import NegamaxAI, Human
from Board import OthelloBoard
from Constants import BLACK, WHITE,EASY,HARD
from RL import QtableAI

#py2exe workaround
import sys
import os
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')


class Othello:

    ##"""
    #Game main class.
    #"""

    def __init__(self):
        """ Show options screen and start game modules"""
        # start
        self.gui = Gui()
        self.board = OthelloBoard()
        self.get_options()        

    def get_options(self):
        # set up players
        player1, player2, mode = self.gui.show_options()
        
        #player1 will always be HUMAN, thus assigning to HUMAN
        self.now_playing = Human(self.gui, BLACK)
        
        if mode == HARD:
            self.other_player = QtableAI(WHITE)
            
            #Load the Qtable from file
            isSuccess = self.other_player.LoadTrainedAI()
            if isSuccess == False:
                self.other_player = NegamaxAI(WHITE, 5)
                
        if mode == EASY:
            self.other_player = NegamaxAI(WHITE, 5)

        self.gui.show_game()
        self.gui.update(self.board.board, 2, 2)

    def run(self):
        
        #clock = pygame.time.Clock()
        while True:
            
            #clock.tick(60)
            isGameOver,winner = self.board.IsGameOver()
            
            if isGameOver == True:
                break
            
            #self.now_playing.get_current_board(self.board)
            moves = self.board.GetValidMoves(self.now_playing.color)
            
            if moves != []:
                
                score, self.board = self.now_playing.MakeMove(self.board)
                
                whites, blacks, empty = self.board.GetCounts()
                
                self.gui.update(self.board.board, blacks, whites)
                                
            self.now_playing, self.other_player = self.other_player, self.now_playing
            
        self.gui.show_winner(winner)
        
        pygame.time.wait(1000)
        
        self.restart()

    def restart(self):
        self.board = OthelloBoard()
        self.get_options()
        self.run()


def main():
    game = Othello()
    game.run()

if __name__ == '__main__':
    main()