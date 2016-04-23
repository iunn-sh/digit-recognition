import sys
import math
import csv

# python showdata.py data/train/ filename 


PIXEL_COUNT = 28


def show(folder,fn ):
    
    with open(folder+fn, 'rb') as f:
        px=f.read(PIXEL_COUNT)
        while(len(px) > 0 ):
            line = ''
            for i in range(len(px)):
                asci = ord(px[i])
                line = '%s%s' %(line,str(asci).ljust(3))
            print line 
            px=f.read(PIXEL_COUNT)  
            
if __name__ == '__main__':
    show(sys.argv[1], sys.argv[2])

