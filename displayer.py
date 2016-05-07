import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
from os import path

# python displayer.py train 04cbb3ce
# python displayer.py test 5926da93

LENGTH = 28


def show_pixel(folder, name):
    script_dir = path.dirname(__file__)

    if folder == 'train':
        pixel_file = 'data/train.csv'
        label_file = 'data/label.csv'
        deskew_pixel_file = 'feature/deskew_data_train.csv'
        contour_file = 'feature/contour_data_train.csv'
        bbox_file = 'feature/bbox_data_train.csv'
    elif folder == 'test':
        pixel_file = 'data/test.csv'
        label_file = 'data/submit.csv'
        deskew_pixel_file = 'feature/deskew_data_test.csv'
        contour_file = 'feature/contour_data_test.csv'
        bbox_file = 'feature/bbox_data_test.csv'
    else:
        print "usage 1: python displayer.py train 001fdcb9"
        print "usage 1: python displayer.py test 00224a98"

    raw = []
    label = []
    deskew = []
    contour = []
    bbox = []
    with open(path.join(script_dir, pixel_file), 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if name == row[0]:
                raw = row
                break
    with open(path.join(script_dir, label_file), 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if name == row[0]:
                label = row
                break
    with open(path.join(script_dir, deskew_pixel_file), 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if name == row[0]:
                deskew = row
                break
    with open(path.join(script_dir, contour_file), 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if name == row[0]:
                contour = row
                break
    with open(path.join(script_dir, bbox_file), 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if name == row[0]:
                bbox = row
                break

    # Two subplots, unpack the axes array immediately
    img_raw = np.array(raw[1:], dtype=np.float)
    img_raw.shape = (LENGTH, LENGTH)
    img_deskew = np.array(deskew[1:], dtype=np.float)
    img_deskew.shape = (LENGTH, LENGTH)
    point_contour = np.array(contour[1:], dtype=np.int)
    point_contour.shape = (-1, 2)
    atr_bbox = np.array(bbox[1:5], dtype=np.int)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img_raw, cmap=plt.cm.gray)
    ax2.imshow(img_deskew, cmap=plt.cm.gray)

    ax2.scatter(x=[coord[0] for coord in point_contour], y=[
                coord[1] for coord in point_contour], c='r')

    bbox_width = atr_bbox[2]
    bbox_height = atr_bbox[3]
    bbox_x = atr_bbox[0] - (0.5 * bbox_width)
    bbox_y = atr_bbox[1] - (0.5 * bbox_height)
    ax2.add_patch(
        patches.Rectangle(
            (bbox_x, bbox_y),
            bbox_width,
            bbox_height,
            alpha=0.3,
            facecolor="#00ffff"
        )
    )

    plt.suptitle('%s' % label)
    plt.show()

if __name__ == "__main__":
    show_pixel(sys.argv[1], sys.argv[2])
