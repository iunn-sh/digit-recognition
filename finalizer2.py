'''python finalizer2.py input output''' 
import sys
import numpy as np
import pandas as pd
from os import path


def finalizer(inputFile,outputFile):
	data = pd.read_csv(path.join(".", inputFile),header=None)
	header = data.iloc[:,0:1].values
	data = data.iloc[:,1:].values
	data = np.where(data > 0.90, 1 , data)
	data = np.where(data < 0.1, 0 , data)
	data = np.hstack((header, data))
	np.savetxt(outputFile, data, delimiter=',', header = '', fmt='%s')

def findNotSureData(inputFile,outputFile):
	data = pd.read_csv(path.join(".", inputFile),header=None)
	data = data.values
	cond1 = (data[:,1]< 0.6) & (data[:,2]< 0.6) & (data[:,3]< 0.6) 
	cond2 = (data[:,4]< 0.6) & (data[:,5]< 0.6) & (data[:,6]< 0.6)
	cond3 = (data[:,7]< 0.6) & (data[:,8]< 0.6) & (data[:,9]< 0.6) & (data[:,10]< 0.6)
	data = data[cond1&cond2&cond3]
	np.savetxt(outputFile, data, delimiter=',', header = '', fmt='%s')       

if __name__ == '__main__':
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    finalizer(inputFile,outputFile)
    #print(len(sys.argv))
    if(len(sys.argv) > 3):
    	outputNotSure = sys.argv[3]
    	findNotSureData(inputFile,outputNotSure)