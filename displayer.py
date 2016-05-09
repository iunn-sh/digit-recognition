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
        binary_pixel_file = 'feature/binary_data_train.csv'
        skeleton_pixel_file = 'feature/skeleton_data_train.csv'
        erosion_pixel_file = 'feature/erosion_data_train.csv'
        contour_file = 'feature/contour_data_train.csv'
        bbox_file = 'feature/bbox_data_train.csv'
    elif folder == 'test':
        pixel_file = 'data/test-new.csv'
        label_file = 'data/submit.csv'
        deskew_pixel_file = 'feature/deskew_data_test-new.csv'
        binary_pixel_file = 'feature/binary_data_test-new.csv'
        skeleton_pixel_file = 'feature/skeleton_data_test-new.csv'
        erosion_pixel_file = 'feature/erosion_data_test-new.csv'
        contour_file = 'feature/contour_data_test-new.csv'
        bbox_file = 'feature/bbox_data_test-new.csv'
    else:
        print "usage 1: python displayer.py train 001fdcb9"
        print "usage 1: python displayer.py test 00224a98"

    raw = []
    label = []
    deskew = []
    binary = []
    skeleton = []
    erosion = []
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
    with open(path.join(script_dir, binary_pixel_file), 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if name == row[0]:
                binary = row
                break
    with open(path.join(script_dir, skeleton_pixel_file), 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if name == row[0]:
                skeleton = row
                break
    with open(path.join(script_dir, erosion_pixel_file), 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if name == row[0]:
                erosion = row
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

    # read data to array
    img_raw = np.array(raw[1:], dtype=np.float)
    img_raw.shape = (LENGTH, LENGTH)
    img_deskew = np.array(deskew[1:], dtype=np.float)
    img_deskew.shape = (LENGTH, LENGTH)
    img_binary = np.array(binary[1:], dtype=np.float)
    img_binary.shape = (LENGTH, LENGTH)
    img_skeleton = np.array(skeleton[1:], dtype=np.float)
    img_skeleton.shape = (LENGTH, LENGTH)
    img_erosion = np.array(erosion[1:], dtype=np.float)
    img_erosion.shape = (LENGTH, LENGTH)
    point_contour = np.array(contour[1:], dtype=np.int)
    point_contour.shape = (-1, 2)
    atr_bbox = np.array(bbox[1:5], dtype=np.int)

    # subplots, unpack the axes array immediately
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True)
    ax1.imshow(img_raw, cmap=plt.cm.gray)
    ax1.set_title('raw')
    ax2.imshow(img_deskew, cmap=plt.cm.gray)
    ax2.set_title('deskew')
    ax3.imshow(img_deskew, cmap=plt.cm.gray)
    ax3.set_title('deskew + contour + bbox')
    ax4.imshow(img_binary, cmap=plt.cm.gray)
    ax4.set_title('deskew + binary')
    ax5.imshow(img_erosion, cmap=plt.cm.gray)
    ax5.set_title('deskew + erosion')
    ax6.imshow(img_skeleton, cmap=plt.cm.gray)
    ax6.set_title('deskew + binary + skeleton')

    ax3.scatter(x=[coord[0] for coord in point_contour], y=[
                coord[1] for coord in point_contour], c='r')

    bbox_width = atr_bbox[2]
    bbox_height = atr_bbox[3]
    bbox_x = atr_bbox[0] - (0.5 * bbox_width)
    bbox_y = atr_bbox[1] - (0.5 * bbox_height)
    ax3.add_patch(
        patches.Rectangle(
            (bbox_x, bbox_y),
            bbox_width,
            bbox_height,
            alpha=0.3,
            facecolor="#00ffff"
        )
    )

    plt.suptitle('%s : %s' % (folder, label))
    plt.show()

if __name__ == "__main__":
    show_pixel(sys.argv[1], sys.argv[2])
