########################################################################

CS510- Image Computation

Avinash Konduru
Nikhila Chireddy

########################################################################

Steps to run the code:
1. Extract the contents of the tar file to the folder which contains the VGG source code.
2. Run the command: python mosse.py <Input video file name> <Output video file name> <CSV file name>

This system takes videos of any input format and generates ouput in .mp4 format. In our experiments we have used .mp4 and .mov
formats for input.

The output .csv consists of seven fields:
->For still objects it stores the - type (still), frame #, frame #, x (upper left), y (upper left), object label, and activation level.
->For moving objects it stores- type (moving), start frame #, end frame #, x (upper left, first frame), y (upper left, first frame),
object label, and percent of frames matching that label. 

#########################################################################
