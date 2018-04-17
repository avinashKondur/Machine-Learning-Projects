# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 14:00:46 2017

@author: Avi
"""
from AIHelpers import Game,Negamax
#from RL import QTable
import random
import copy

class Human:

    def __init__(self, gui, color="black"):
        
        self.color = color
        self.gui = gui

    def MakeMove(self,board):

        """ Uses gui to handle mouse
        """
        validMoves = board.GetValidMoves(self.color)
        
        while True:
            move = self.gui.get_mouse_input()
            if move in validMoves:
                break
            
        board.MakeMove(move, self.color)
        return 0, board



class NegamaxAI(object):

    def __init__(self, color, depth=3):
        self.depthLimit = depth
        self.negMax = Negamax(self.depthLimit)
        self.color = color
        self.ebf = []

    def MakeMove(self,board):
        
        #initalise the with state of current board to explore the moves
        game = Game(board,self.color)        
        
        #get the best move using NegamaxIDSab
        value,move = self.negMax.NegamaxIDSab(game)
        
        #append the ebf value curent move 
        #self.ebf.append(game.getEbf())
        
        if move is None:
            move = random.sample(game.GetMoves(), 1)[0]
            
        board.MakeMove(move,self.color)
        
        return value,board
    
    # Used in Reinforcement Learning
    def GetMove(self, board):
        game = Game(board, self.color)
        value,move = self.negMax.NegamaxIDSab(game)
        if move is None:
            if game.GetMoves() != []:
                return random.sample(game.GetMoves(), 1)[0]
            else:
                return []
        else:
            return move
        


    
class RandomPlayer (NegamaxAI):

    def get_move(self,board):
        
        game = Game(board,self.color)
        
        x = random.sample(game.getMoves(), 1)
        
        return x[0]