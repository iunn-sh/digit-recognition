from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from numpy import genfromtxt, savetxt, std, argsort
from os import path

def create_forest():
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
    forest = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1)
    forest.fit(train, target)
    prob = forest.predict_proba(test)
    savetxt(path.join(script_dir, 'data/sample.csv'),
            prob, delimiter=',', fmt='%f')

    return forest

def get_oob(forest):
    print "out-of-bag accuracy = ", forest.oob_score_
    return forest.oob_score_


if __name__ == "__main__":
    forest = create_forest()
    get_oob(forest)
