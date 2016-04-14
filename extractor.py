import sys
import math
import csv
import numpy as np
import recognizer

# python extractor.py data/label.csv data/train.csv data/train/
# python extractor.py data/sample.csv data/test.csv data/test/

LENGTH = 28
PIXEL_COUNT = LENGTH * LENGTH

def get_raw(fn):
    with open(fn, 'rb') as f:
        px=f.read(PIXEL_COUNT)
        line = []
        for i in range(PIXEL_COUNT):
            line.append(ord(px[i]))

    return line

def gen_xy_sum(output_fn):
    pixel = recognizer.read_csv_to_list(output_fn)
    xy_sum = []

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)

        x_sum = []
        y_sum = []
        for i in range(LENGTH):
            x_list = img[i]
            x_sum.append(str(sum(1 if x>0 else 0 for x in x_list)))
            y_list = []
            for j in range(LENGTH):
                y_list.append(img[j, i])
            y_sum.append(str(sum(1 if y>0 else 0 for y in y_list)))
        # print "xy_sum length:", len(x_sum), len(y_sum)

        linked = [name] + x_sum + y_sum
        # print linked
        xy_sum.append(linked)
    with open('feature/xy_sum_'+output_fn.replace("/","_"), 'wb') as f:
        for l in xy_sum:
            f.write(','.join(l)+'\n')

def get_csv(fn, output_fn, folder):
    feas=[]
    with open(fn, 'rb') as f:
        # for each file in sample file
        for l in csv.reader(f):
            fea=get_raw(folder+l[0])
            # for training data, with label
            # feas.append([l[0], l[1]]+map(str, fea))
            # for testing data, without label
            feas.append([l[0]]+map(str, fea))

    with open(output_fn, 'wb') as f:
        # write features of each file into csv
        for l in feas: f.write(','.join(l)+'\n')

if __name__ == '__main__':
    get_csv(sys.argv[1], sys.argv[2], sys.argv[3])
    gen_xy_sum(sys.argv[2])
