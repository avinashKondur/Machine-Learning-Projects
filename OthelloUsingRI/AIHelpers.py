# -*- coding: utf-8 -*-
#"""
#Created on Sat Dec 09 13:49:13 2017

#This class is used for AI which takes the curent state of game 
#and explores the move to return the better move

#@author: Avi
#"""
from Constants import WHITE, BLACK
import copy

class Game(object):

    def __init__(self, board,player):

        #initialise board for the game        
        self.board = copy.deepcopy(board)
        
        #initialise the starting player
        self.player = player
        
        #look ahead player used for AI
        self.playerLookAHead = self.player
        
        #set opponents color
        self.opponent = BLACK if self.player == WHITE else WHITE
        
        self.movesExplored = 0


    def GetMoves(self):
        moves = self.board.GetValidMoves(self.playerLookAHead)
        return moves

    def GetUtility(self):
        
        isGameOver,player_won = self.board.IsGameOver()
        
        if isGameOver == True:
            if player_won == WHITE:
                return 1 if self.player is WHITE else -1
            elif player_won == BLACK:
                return 1 if self.player is BLACK else -1
            else:
                return 0
        else:
            return None
        
    def IsOver(self):
        return self.GetUtility() is not None

    def makeMove(self, move):
        self.board.MakeMove(move, self.playerLookAHead)        
        self.playerLookAHead = WHITE if self.playerLookAHead == BLACK else BLACK
        self.movesExplored +=1

    def unmakeMove(self, move):
        self.board.UnmakeMove()
        self.playerLookAHead = WHITE if self.playerLookAHead == BLACK else BLACK
    
    def getWinningValue(self):
        return 1
    
    def getMovesExplored(self):
        return self.movesExplored
    
    
    def getDepth(self):
        whites,blacks,blanks = self.board.GetCounts()
        return whites + blacks
    
    def __ebf(self,nNodes, depth, precision=0.01):
        if nNodes == 0:
            return 0

        def __ebfRec(low, high):
            mid = (low + high) * 0.5
            if mid == 1:
                estimate = 1 + depth
            else:
                estimate = (1 - mid**(depth + 1)) / (1 - mid)
            if abs(estimate - nNodes) < precision:
                return mid
            if estimate > nNodes:
                return __ebfRec(low, mid)
            else:
                return __ebfRec(mid, high)

        return __ebfRec(1, nNodes)

    def getEbf(self):
        return self.__ebf(self.getMovesExplored(),self.getDepth())
    
    def __str__(self):
        s = '{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}'.format(*self.board)
        return s
    
class Negamax:
    
    def __init__(self, depth):
                
        #initialise the depth
        self.depthLimit = depth
    
    def __negamax(self,game, depthLeft, alpha = None, beta = None):
        
        #print('IS Game over = ',game.IsOver(),depthLeft)
        
        isGameOver = game.IsOver()
        
        # If at terminal state or depth limit, return utility value and move None
        if isGameOver == True or depthLeft == 0:
            return game.GetUtility(), None # call to negamax knows the move
        # Find best move and its value from current state
        bestValue, bestMove = None, None
        
        validMoves = game.GetMoves()
        
        for move in validMoves:
            
            #prevBoard = copy.deepcopy(game.board.board)
            
            # Apply a move to current state
            game.makeMove(move)
            
            if alpha is not None and beta is not None:
                alpha = - alpha
                beta = - beta
                
            # Use depth-first search to find eventual utility value and back it up.
            #  Negate it because it will come back in context of next player
            value, _ = self.__negamax(copy.deepcopy(game), depthLeft-1, beta, alpha)
            
            # Remove the move from current state, to prepare for trying a different move
            game.unmakeMove(move)
            #game.board.board = prevBoard
            
            if value is None:
                continue
            value = - value
            if bestValue is None or value > bestValue:
                # Value for this move is better than moves tried so far from this state.
                bestValue, bestMove = value, move
                
            if alpha is not None and beta is not None:
                if bestValue >= beta:
                    return bestValue, bestMove
                alpha = max(bestValue, alpha)
            
        return bestValue, bestMove

    def NegamaxIDSab(self,game):
        
        alpha = - float('inf')
        beta = float('inf')
        
        bestValue, bestMove = None, None
        for depth in range(self.depthLimit):
            value,move = self.__negamax(game, depth,alpha,beta)
            
            if bestValue is None or value > bestValue:
                bestValue, bestMove = value, move
                
            if bestValue is game.getWinningValue():
                return bestValue, bestMove
            
        return bestValue, bestMove

