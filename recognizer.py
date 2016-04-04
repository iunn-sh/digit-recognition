from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from os import path
import matplotlib.pyplot as plt

def main():
    script_dir = path.dirname(__file__) #<-- absolute dir the script is in

    # create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open(path.join(script_dir, 'data/train.csv'), 'r'),
                             delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open(path.join(script_dir, 'data/test.csv'), 'r'), delimiter=',', dtype='f8')[1:]

    # create and train the random forest
    # multi-core CPUs can use: rf =
    # RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    rf.fit(train, target)
    savetxt(path.join(script_dir, 'data/submit.csv'), rf.predict_proba(test), delimiter=',', fmt='%f')


if __name__ == "__main__":
    main()
