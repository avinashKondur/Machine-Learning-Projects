from Board import OthelloBoard
from Constants import WHITE, BLACK, DRAW
import numpy as np
import copy
from Players import NegamaxAI
import glob

class QTable:

    def __init__(self):
        self.Q = {}
        self.rho = 0.2
        self.epsilonDecayRate = 0.9999
        self.epsilon = 1

    def __moveColorTuple(self, move, color):
        mc = []                
        mc.append(tuple(move))
        mc.append(color)        
        return tuple(mc)
        
    def __makeTuple(self, bd):
        l = []
        for row in bd:
            l.append(tuple(row))
        return tuple(l)
        
    def __boardMoveTuple(self,makeT,moveColor):
        return str((makeT,moveColor))
            
    def LoadQTable(self):
        
        file = glob.glob('./trained.qtable')
        
        if file == []:
            print "No Trained Q table found"
            return False
        
        fr = open(file[0],'r')        
        data = fr.read()
        
        if data != '':
            lines = data.split('\n')
            for line in lines:
                #print line
                keyVal = line.split('>>>')
                if len(keyVal) == 2:
                    self.Q[keyVal[0].strip()] = float(keyVal[1].strip())
            
            print "Q table loading success"
            fr.close()
            return True
        else:
            print "No data found in Q table"
            fr.close()
            return False
        
        
        
        
    def SaveQtable(self):
        
        fout = "trained.qtable"
        fo = open(fout, "w")

        for k, v in self.Q.items():
            fo.write(str(k) + ' >>> '+ str(v) + '\n')

        fo.close()            
    
    def UpdateQTable(self,boardOld,moveOld,board,move, firstMove, isGameOver,winDisk ):
        
        #isGameOver, winDisk = board.IsGameOver()
        
        if self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE)) not in self.Q:
            self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] = 0
            
        if isGameOver == True: 
            if winDisk == WHITE:
                # WHITE won!
                self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] = 1                                        
            if winDisk == BLACK:
                self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] += self.rho * (-1 - self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))])                        
            if winDisk == DRAW:
                # Game over. No winner.
                self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] = 0
        else:
            if firstMove == False:
                self.Q[self.__boardMoveTuple(self.__makeTuple(boardOld.board),self.__moveColorTuple(moveOld, WHITE))] += self.rho * ( self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] - self.Q[self.__boardMoveTuple(self.__makeTuple(boardOld.board),self.__moveColorTuple(moveOld, WHITE))])
                        

    
    def GetMove(self, board,color,epsilon = None):
        validMoves = board.GetValidMoves(color)
        if validMoves == []:
              return []
        if epsilon != None and np.random.uniform() < epsilon:
              # Random Move
                 return validMoves[ np.random.choice(len(validMoves))]
        else:
            # Greedy Move
            Qs = np.array([self.Q.get(self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, color)), 0) for move in validMoves]) 
            return validMoves[ np.argmax(Qs) ]
            
    def Train(self, maxGames, rho, epsilonDecayRate, epsilon):
        success = self.LoadQTable()
        if success == False:
            self.Q = {}
        
        outcomes = np.zeros(maxGames)
        epsilons = np.zeros(maxGames)
        steps = np.zeros(maxGames)
        
        # opponent plays using negamax
        opponent = NegamaxAI(BLACK,5)
        
        for nGames in range(maxGames):
            epsilon *= epsilonDecayRate
            epsilons[nGames] = epsilon
            step = 0
            board = OthelloBoard()
            #board = bd.board  # empty board
            done = False
            boardOld,moveOld = None, None
            
            print "Training AI for {0} time".format(nGames)
    
            while not done:        
                step += 1
        
                # WHITE's turn
                move = self.GetMove(board, WHITE,epsilon)
                if move == ():
                    print "AI move is empty"
                #print(move)
                if move != ():
                    boardNew = copy.deepcopy(board)
                    #print((self.__makeTuple(boardNew.board),self.__moveColorTuple(move, WHITE)))
                    
                    if self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE)) not in self.Q:
                        self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] = 0  # initial Q value for new board,move
                        
                    boardNew.MakeMove(move, WHITE)
                
                isGameOver, winDisk = boardNew.IsGameOver()
                if isGameOver == True: 
                    done = True
                    if winDisk == WHITE:
                        # WHITE won!
                        self.Q[(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] = 1                        
                        outcomes[nGames] = 1
                    if winDisk == BLACK:
                        self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] += rho * (-1 - self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))])
                        outcomes[nGames] = -1            
                    if winDisk == DRAW:
                        # Game over. No winner.
                        self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] = 0
                        outcomes[nGames] = 0            
                else:
                    moveB = opponent.GetMove(boardNew)
                    
                    #BLACK's turn. BLACK is a random player
                    #validMoves = boardNew.GetValidMoves(BLACK)
                    if moveB != []:
                        #moveB = validMoves[ np.random.choice(len(validMoves))]
                        boardNew.MakeMove(moveB, BLACK)
                        isGameOver, winDisk = boardNew.IsGameOver()
                        if isGameOver == True:
                            # BLACK won!
                            done = True
                            if winDisk == WHITE:
                                # WHITE won!
                                self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] = 1                        
                                outcomes[nGames] = 1
                            if winDisk == BLACK:
                                self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] += rho * (-1 - self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))])
                                outcomes[nGames] = -1            
                            if winDisk == DRAW:
                                # Game over. No winner.
                                self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] = 0
                                outcomes[nGames] = 0
        
                if step > 1:
                    #update old += rho *(-1 qnew - qold)
                    self.Q[self.__boardMoveTuple(self.__makeTuple(boardOld.board),self.__moveColorTuple(moveOld, WHITE))] += rho * ( self.Q[self.__boardMoveTuple(self.__makeTuple(board.board),self.__moveColorTuple(move, WHITE))] - self.Q[self.__boardMoveTuple(self.__makeTuple(boardOld.board),self.__moveColorTuple(moveOld, WHITE))])
            
                boardOld, moveOld = copy.deepcopy(board), copy.deepcopy(move) # remember board and move to Q(board,move) can be updated after next steps
                board = copy.deepcopy(boardNew)
            
            steps[nGames] = step
            self.SaveQtable()
                
        #return self.Q
        return outcomes,epsilons,steps
        
class QtableAI(object):
    
    def __init__(self, color):
        self.color = color
        self.Qtable  = QTable()
        self.FirstMove = True
        self.boardOld = None
        self.moveOld = None
    
    def LoadTrainedAI(self):
        success = self.Qtable.LoadQTable()
        return success
        
    def MakeMove(self, board):
        
        boardBeforeMove = copy.deepcopy(board)
        move = self.Qtable.GetMove(board,self.color)
        
        #make move
        board.MakeMove(move,self.color)
        
        isGameOver, winDisk = board.IsGameOver()
        
        self.Qtable.UpdateQTable(self.boardOld,self.moveOld,boardBeforeMove,move,self.FirstMove,isGameOver,winDisk)
        
        if self.FirstMove == True:
            self.FirstMove = False
        
        self.boardOld = boardBeforeMove
        self.moveOld = move
        
        self.Qtable.SaveQtable()
        
        return 0,board
          

def main():
    qtable = QTable()
    outcomes,epsilons,steps = qtable.Train(100, 0.2, 0.9999, 1.0)
    
    #qtable.SaveQtable()
    
    np.savetxt('Outcomes.txt',outcomes)
    np.savetxt('epsilons.txt',epsilons)
    np.savetxt('steps.txt',steps)
       
if __name__ == '__main__':
    main()