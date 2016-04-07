from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from numpy import genfromtxt, savetxt
from os import path

def random_forest():
    # create the training & test sets, skipping the header row with [1:]
    script_dir = path.dirname(__file__)  # <-- absolute dir the script is in
    dataset = genfromtxt(open(path.join(script_dir, 'data/train.csv'), 'r'),
                            delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open(path.join(script_dir, 'data/test.csv'), 'r'),
                        delimiter=',', dtype='f8')[1:]

    # create and train the random forest
    # multi-core CPUs can use: rf =
    # RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1)
    rf.fit(train, target)
    prob = rf.predict_proba(test)
    savetxt(path.join(script_dir, 'data/sample.csv'),
            prob, delimiter=',', fmt='%f')
    print "out-of-bag accuracy = ", rf.oob_score_
    return rf.oob_score_

    # score = log_loss(target, prob)
    # print score

if __name__ == "__main__":
    random_forest()
