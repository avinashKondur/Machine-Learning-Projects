################################################################

Authors : Avinash Konduru 
	  Nikhila Chireddy

################################################################

Pre-requisites:

	This game works only with Python Version - 2.7

	Ensure that the Pygame library is installed.

	If not installed, please use the command:

		>conda install -c cogsci pygame

To run the game:

	>python OthelloGame.py

Once this command is run, a Pygame window will be opened. There are two options to select.

1. Negamax AI - The AI internally uses the negamax algorithm in this mode.
2. RL QTable AI - The AI uses Reinforcement Learning in this mode.

Once the game is finished, the winner is displayed and a new game starts in 10 seconds.