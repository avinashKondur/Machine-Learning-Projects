import csv

def CreateFile(filename):
	with open(filename, 'a') as csvfile:
			fieldnames = ['StartFrame#', 'EndFrame#','X(UL)','Y(UL)','ObjectLabel','ActivationLevel(probability)','ObjectType']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()


def writeOutput(filename,s,e,xl,yl,ol,al,otype):
	with open(filename, 'a') as csvfile:
		fieldnames = ['StartFrame#', 'EndFrame#','X(UL)','Y(UL)','ObjectLabel','ActivationLevel(probability)','ObjectType']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writerow({'StartFrame#': s, 'EndFrame#': e ,'X(UL)': xl ,'Y(UL)': yl ,'ObjectLabel': ol ,'ActivationLevel(probability)': al ,'ObjectType':otype})
		
#writeOutput(1,1,2,3,1,3)
#writeOutput(1,2,4,1,5,6)