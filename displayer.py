import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
from os import path

# python displayer.py train 001fdcb9
# python displayer.py test 00224a98

def show_pixel(folder, name):
    script_dir = path.dirname(__file__)

    raw = []
    label = []
    if folder == 'train':
        with open(path.join(script_dir, 'data/train.csv'), 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if name == row[0]:
                    raw = row
                    break
        with open(path.join(script_dir, 'data/label.csv'), 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if name == row[0]:
                    label = row
                    break
    elif folder == 'test':
        with open(path.join(script_dir, 'data/test.csv'), 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if name == row[0]:
                    raw = row
                    break
        with open(path.join(script_dir, 'data/submit.csv'), 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if name == row[0]:
                    label = row
                    break
    else:
        print "usage 1: python displayer.py train 001fdcb9"
        print "usage 1: python displayer.py test 00224a98"

    img = np.array(raw[1:], dtype=np.float)
    img.shape = (28, 28)
    imgplot = plt.imshow(img, cmap=plt.cm.gray)
    plt.title('%s' %label)
    plt.show()

if __name__ == "__main__":
    show_pixel(sys.argv[1], sys.argv[2])
