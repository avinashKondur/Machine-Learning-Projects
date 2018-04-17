import copy

from Constants import WHITE, BLACK, DRAW

class OthelloBoard:
    
    def __init__(self):
        self.board = [[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]]
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[3][3] = WHITE
        self.board[4][4] = WHITE
        self.validMoves = []
        self.prevState = copy.deepcopy(self.board)
        
    def __getDisk__(self, i, j):
        return self.board[i][j]
    
    def __possibleMoves(self, row, col, color):
        '''Given the color and position of disk, it returns the possible positions such that there exists atleast one stright
        line between the given disk and another disk of same color'''
        
        if color == BLACK:
            opponent = WHITE
        else:
            opponent = BLACK
        
        positions = []
        
        if (row < 0 or row > 7 or col < 0 or col > 7):
            return positions
        
        #Check for possible positions in each direction.
        
        #north
        i = row-1
        if (i >= 0 and self.board[i][col] == opponent):
            i = i - 1
            while (i >= 0 and self.board[i][col] == opponent):
                i = i - 1
            if (i >= 0 and self.board[i][col] == 0):
                positions = positions + [(i, col)]
                
        
        #south
        i = row+1
        if (i <= 7 and self.board[i][col] == opponent):
            i = i + 1
            while (i <= 7 and self.board[i][col] == opponent):
                i = i + 1
            if (i <= 7 and self.board[i][col] == 0):
                positions = positions + [(i, col)]
                
        
        #west
        j = col-1
        if (j >= 0 and self.board[row][j] == opponent):
            j = j - 1
            while (j >= 0 and self.board[row][j] == opponent):
                j = j - 1
            if (j >= 0 and self.board[row][j] == 0):
                positions = positions + [(row, j)]
                
                
        #east
        j = col+1
        if (j <= 7 and self.board[row][j] == opponent):
            j = j + 1
            while (j <= 7 and self.board[row][j] == opponent):
                j = j + 1
            if (j <= 7 and self.board[row][j] == 0):
                positions = positions + [(row, j)]
                
        
        #north-west
        i = row-1
        j = col-1
        if(i >= 0 and j >= 0 and self.board[i][j] == opponent):
            i = i - 1
            j = j - 1
            while(i >= 0 and j >= 0 and self.board[i][j] == opponent):
                i = i - 1
                j = j - 1
            if(i >= 0 and j >= 0 and self.board[i][j] == 0):
                positions = positions + [(i, j)]
        
        
        #north-east
        i = row-1
        j = col+1
        if(i >= 0 and j <= 7 and self.board[i][j] == opponent):
            i = i - 1
            j = j + 1
            while(i >= 0 and j <= 7 and self.board[i][j] == opponent):
                i = i - 1
                j = j + 1
            if(i >= 0 and j <= 7 and self.board[i][j] == 0):
                positions = positions + [(i, j)]
                
                
        #south-west
        i = row+1
        j = col-1
        if(i <= 7 and j >= 0 and self.board[i][j] == opponent):
            i = i + 1
            j = j - 1
            while(i <= 7 and j >= 0 and self.board[i][j] == opponent):
                i = i + 1
                j = j - 1
            if(i <= 7 and j >= 0 and self.board[i][j] == 0):
                positions = positions + [(i, j)]
                
                
        #south-east
        i = row+1
        j = col+1
        if(i <= 7 and j <= 7 and self.board[i][j] == opponent):
            i = i + 1
            j = j + 1
            while(i <= 7 and j <= 7 and self.board[i][j] == opponent):
                i = i + 1
                j = j + 1
            if(i <= 7 and j <= 7 and self.board[i][j] == 0):
                positions = positions + [(i, j)]
                
                
        return positions
    
    
    def GetValidMoves(self, color):
        '''return the set of possible moves for a given disk color'''
        
        moves = []
        
        for i in range (0,8):
            for j in range(0,8):
                if self.board[i][j] == color:
                    moves = moves + self.__possibleMoves(i, j, color)
                    
        moves = list(set(moves))
        self.validMoves = moves
        return moves
    
    
    def MakeMove(self, move, color):
        '''applies the move for the specified color and makes changes to the board'''
        
        self.prevState = copy.deepcopy(self.board)
        
        if move in self.validMoves:
            self.board[move[0]][move[1]] = color
            for direction in range(0,8):
                self.Flip(direction, move, color)
            
            
            
    def Flip(self, direction, position, color):
        '''Flips the disks of given color in the given position'''
        
        #north
        if direction == 0:
            row = -1
            col = 0
        #south
        elif direction == 1: 
            row = 1
            col = 0
        #west
        elif direction == 2:
            row = 0
            col = -1
        #east
        elif direction == 3:
            row = 0
            col = 1
        #north west
        elif direction == 4:
            row = -1
            col = -1
        #north east
        elif direction == 5:
            row = -1
            col = 1
        #south west
        elif direction == 6:
            row = 1
            col = -1
        #south east
        elif direction == 7:
            row = 1
            col = 1
            
            
        i = position[0] + row
        j = position[1] + col
        
        flipDisks = []
        
        if color is BLACK:
            opponent = WHITE
        else:
            opponent = BLACK
        
        if i in range(0, 8) and j in range(0, 8) and self.board[i][j] == opponent:
            flipDisks = flipDisks + [(i, j)]
            i = i + row
            j = j + col
            #repeatedly increment till the color appears
            while i in range(0, 8) and j in range(0, 8) and self.board[i][j] == opponent:
                flipDisks = flipDisks + [(i, j)]
                i = i + row
                j = j + col
            
            #check if the color appeared. If yes, then flip the disks. 
            if i in range(0, 8) and j in range(0, 8) and self.board[i][j] == color:
                for disk in flipDisks:
                    self.board[disk[0]][disk[1]] = color
                    
            
    def GetCounts(self):
        '''Get the number of whites, blacks and blank spaces on the board'''
        
        whites = 0
        blacks = 0
        blanks = 0
        
        for i in range(0, 8):
            for j in range(0, 8):
                if self.board[i][j] == WHITE:
                    whites = whites + 1
                elif self.board[i][j] == BLACK:
                    blacks = blacks + 1
                else:
                    blanks = blanks + 1
                    
        return whites, blacks, blanks
    
    def IsGameOver(self):
        
        whites, blacks, blanks = self.GetCounts()
        maxDisks = WHITE
        if whites < blacks:
            maxDisks = BLACK
        elif whites == blacks:
            maxDisks = DRAW
        #if there are no whites or blacks on the board, or if the board is full
        if whites == 0 or blacks == 0 or blanks == 0:
            return True, maxDisks
        
        #if there are no more moves for both the players
        whiteMoves = self.GetValidMoves(WHITE)
        blackMoves = self.GetValidMoves(BLACK)
        
        if whiteMoves == [] and blackMoves == []:
            return True, maxDisks
        return False, None
    
    def PrintBoard(self):
        
        for i in range(0, 8):
            print i,' |',
            
            for j in range(0, 8):
                if self.board[i][j] == WHITE:
                    print 'W',
                elif self.board[i][j] == BLACK:
                    print 'B',
                else:
                    print ' ',
                print ' |',
            print
            
    
    def UnmakeMove(self):
        self.board = copy.deepcopy(self.prevState)
        