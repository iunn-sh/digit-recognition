import sys
import math
import csv
import numpy as np
import cv2
import random_forest

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

def gen_deskew(output_fn):
    pixel = random_forest.read_csv_to_list(output_fn)
    deskew = []
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img_deskew = []
        # print "img:", len(img), img

        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            img_deskew = img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*LENGTH*skew], [0, 1, 0]])
        img_deskew = cv2.warpAffine(img,M,(LENGTH, LENGTH),flags=affine_flags)
        # print "img_deskew:", len(img_deskew.flatten()), img_deskew.flatten()

        linked = [name] + ['{}'.format(x) for x in img_deskew.flatten()]
        # print linked
        deskew.append(linked)
    with open('feature/deskew_'+output_fn.replace("/","_"), 'wb') as f:
        for l in deskew:
            f.write(','.join(l)+'\n')


def gen_xy_sum(output_fn):
    pixel = random_forest.read_csv_to_list(output_fn)
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
    gen_deskew(sys.argv[2])
    gen_xy_sum(sys.argv[2])
