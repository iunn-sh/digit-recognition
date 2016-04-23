'''python finalizer2.py input output''' 
import sys
import numpy as np
import pandas as pd
from os import path


def finalizer(inputFile,outputFile):
	data = pd.read_csv(path.join(".", inputFile),header=None)
	header = data.iloc[:,0:1].values
	data = data.iloc[:,1:].values
	data = np.where(data > 0.99, 1 , data)

	#data = np.where(data < 0.1, 0 , data)
	data = np.hstack((header, data))

	np.savetxt(outputFile, data, delimiter=',', header = '', fmt='%s')




#
# less than , and . all element less than 0.6 
# large than, or .  any element large than 0.9
#
def getCond(data, oper,prob):
	
	if (oper == "<"):
		cond = (data[:,1] <= prob)
		for i in range(2,11):
			cond = cond & (data[:,i] <= prob)
	else:
		cond = (data[:,1] >= prob)
		for i in range(2,11):
			cond = cond | (data[:,i] >= prob)
	return cond

def findNotSureData(inputFile,outputFile):
	data = pd.read_csv(path.join(".", inputFile),header=None)
	data = data.values
	 
	data = data[getCond(data,"<",0.99)]
	data = data[getCond(data,">",0.90)]
	np.savetxt(outputFile, data, delimiter=',', header = '', fmt='%s')       

if __name__ == '__main__':
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    finalizer(inputFile,outputFile)
    #print(len(sys.argv))
    if(len(sys.argv) > 3):
    	outputNotSure = sys.argv[3]
    	findNotSureData(inputFile,outputNotSure)