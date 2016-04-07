import sys
import math
import csv

# python gen-feature.py train.csv feature-train.csv train/

def gen_feature(fn):
    # symmetric in x axis
    x_sym=0
    x_sym_cnt=0
    # symmetric in y axis
    y_sym=0
    y_sym_cnt=0
    # pixel sum/count
    px_sum=0
    px_cnt=0
    # min/max point in x axis
    min_x=28
    max_x=-1
    # min/max point in y axis
    min_y=28
    max_y=-1

    with open(fn, 'rb') as f:
        px=f.read(28*28)
        # compute symmetric in x axis
        for i in xrange(28):
            for j in xrange(14):
                d=ord(px[i*28+j])-ord(px[i*28+(27-j)])
                x_sym+=float(abs(d))/256
                if abs(d)>0: x_sym_cnt+=1
        # compute symmetric in y axis
        for i in xrange(14):
            for j in xrange(28):
                d=ord(px[i*28+j])-ord(px[(27-i)*28+j])
                y_sym+=float(abs(d))/256
                if abs(d)>0: y_sym_cnt+=1
        # compute pixel sum/count
        for i in xrange(len(px)): 
            px_sum+=ord(px[i])
            if ord(px[i])>0: px_cnt+=1
        # find min/max position in x/y axis
        for i in xrange(28):
            for j in xrange(28):
                if ord(px[i*28+j])==0: continue
		if i<min_x: min_x=i
		if i>max_x: max_x=i
		if j<min_y: min_y=j
		if j>max_y: max_y=j
    
    return [x_sym, x_sym/(28*28), x_sym_cnt, float(x_sym_cnt)/(28*28), y_sym, y_sym/(28*28), y_sym_cnt, float(y_sym_cnt)/(28*28), px_sum, float(px_sum)/(28*28), px_cnt, float(px_cnt)/(28*28), max_x, min_x, max_y, min_y, float(max_x-min_x)/(max_y-min_y)]

def get_csv(fn, output_fn, folder):
    feas=[]
    with open(fn, 'rb') as f:
        # for each file in sample file
        for l in csv.reader(f):
            fea=gen_feature(folder+l[0])
            # for training data, with label
            feas.append([l[0], l[1]]+map(str, fea))
            # for testing data, without label
            #feas.append([l[0]]+map(str, fea))

    with open(output_fn, 'wb') as f:
        # write features of each file into csv
        for l in feas: f.write(','.join(l)+'\n')

if __name__ == '__main__':
    get_csv(sys.argv[1], sys.argv[2], sys.argv[3])
