import sys
import math
import csv

# python extractor.py data/label.csv data/train.csv data/train/
# python extractor.py data/sample.csv data/test.csv data/test/

PIXEL_COUNT = 28 * 28

def gen_feature(fn):
    with open(fn, 'rb') as f:
        px=f.read(PIXEL_COUNT)
        line = []
        for i in range(PIXEL_COUNT):
            line.append(ord(px[i]))

    return line

def get_csv(fn, output_fn, folder):
    feas=[]
    with open(fn, 'rb') as f:
        # for each file in sample file
        for l in csv.reader(f):
            fea=gen_feature(folder+l[0])
            # for training data, with label
            # feas.append([l[0], l[1]]+map(str, fea))
            # for testing data, without label
            feas.append([l[0]]+map(str, fea))

    with open(output_fn, 'wb') as f:
        # write features of each file into csv
        for l in feas: f.write(','.join(l)+'\n')

if __name__ == '__main__':
    get_csv(sys.argv[1], sys.argv[2], sys.argv[3])
