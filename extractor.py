import sys
import math
import csv
import numpy as np
import cv2
import random_forest

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure

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

def gen_bounding_box(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
    bounding_box = []

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img = img.astype(np.uint8)

        B = np.argwhere(img)
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
        width = xstop - xstart
        height = ystop - ystart
        center_x = (xstop + xstart) / 2
        center_y = (ystop + ystart) / 2
        ratio = float(width) / float(height)

        flatten = [center_x, center_y, width, height, ratio]
        # print flatten
        linked = [name] + ['{}'.format(x) for x in flatten]
        # print linked
        bounding_box.append(linked)

    with open(output_fn, 'wb') as f:
        for l in bounding_box:
            f.write(','.join(l)+'\n')

def gen_contour(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
    contour = []

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img = img.astype(np.uint8)
        # print img

        ret, thresh = cv2.threshold(img, 127, 255, 0)
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        # print str(len(cnts)),"contours"
        for c in cnts:
            # print str(len(c)), "points"
            cv2.drawContours(img, c, -1, (0,255,0), 3)

        flatten = [val for lists in cnts for sublists in lists for sublist in sublists for val in sublist]
        # print flatten
        linked = [name] + ['{}'.format(x) for x in flatten]
        # print linked
        contour.append(linked)

    with open(output_fn, 'wb') as f:
        for l in contour:
            f.write(','.join(l)+'\n')

# def gen_hog(output_fn):
#     pixel = random_forest.read_csv_to_list(output_fn)
#     hog_pixel = []
#
#     for row in pixel:
#         name = row[0]
#         img = np.array(row[1:], dtype=np.float)
#         img.shape = (LENGTH, LENGTH)
#
#         hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
#                             cells_per_block=(1, 1), visualise=True)
#         print hog_image
#
#         linked = [name] + hog_image
#         hog_pixel.append(linked)
#     with open('feature/hog_'+output_fn.replace("/","_"), 'wb') as f:
#         for l in hog_pixel:
#             f.write(','.join(l)+'\n')

def gen_deskew(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
    deskew = []
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

    for row in pixel:
        name = row[0]
        img = np.array(row[1:], dtype=np.float)
        img.shape = (LENGTH, LENGTH)
        img_deskew = []

        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            img_deskew = img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*LENGTH*skew], [0, 1, 0]])
        img_deskew = cv2.warpAffine(img,M,(LENGTH, LENGTH),flags=affine_flags)

        linked = [name] + ['{}'.format(x) for x in img_deskew.flatten()]
        deskew.append(linked)

    with open(output_fn, 'wb') as f:
        for l in deskew:
            f.write(','.join(l)+'\n')

def gen_xy_sum(fn, output_fn):
    pixel = random_forest.read_csv_to_list(fn)
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

        linked = [name] + x_sum + y_sum
        # print linked
        xy_sum.append(linked)

    with open(output_fn, 'wb') as f:
        for l in xy_sum:
            f.write(','.join(l)+'\n')

def get_csv(fn, output_fn, folder):
    feas=[]
    with open(fn, 'rb') as f:
        # for each file in sample file
        for l in csv.reader(f):
            fea=get_raw(folder+l[0])
            feas.append([l[0]]+map(str, fea))

    with open(output_fn, 'wb') as f:
        # write features of each file into csv
        for l in feas: f.write(','.join(l)+'\n')

if __name__ == '__main__':
    fn = sys.argv[1]
    output_fn = sys.argv[2]
    folder = sys.argv[3]
    get_csv(fn, output_fn, folder)

    output_fn_deskew = 'feature/deskew_'+output_fn.replace("/","_")
    gen_deskew(output_fn, output_fn_deskew)

    output_fn_xy_sum = 'feature/xy_sum_'+output_fn.replace("/","_")
    gen_xy_sum(output_fn_deskew, output_fn_xy_sum)

    output_fn_contour = 'feature/contour_'+output_fn.replace("/","_")
    gen_contour(output_fn_deskew, output_fn_contour)

    output_fn_bbox = 'feature/bbox_'+output_fn.replace("/","_")
    gen_bounding_box(output_fn_deskew, output_fn_bbox)
