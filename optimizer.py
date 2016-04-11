import matplotlib.pyplot as plt
import recognizer

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from numpy import genfromtxt, savetxt
from os import path

RANDOM_STATE = 123

def calculate_oob_error():
    # create the training & test sets, skipping the header row with [1:]
    script_dir = path.dirname(__file__)  # <-- absolute dir the script is in
    # dataset = genfromtxt(open(path.join(script_dir, 'data/train.csv'), 'r'),
    #                         delimiter=',', dtype='f8')[1:]
    target = recognizer.read_csv_to_list(path.join(script_dir, 'data/label.csv'))
    train = recognizer.read_csv_to_list(path.join(script_dir, 'data/train.csv'))
    target_train = recognizer.link_by_key(target, train)

    # NOTE: Setting the `warm_start` construction parameter to `True` disables
    # support for paralellised ensembles but is necessary for tracking the OOB
    # error trajectory during training.
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(warm_start=True, oob_score=True,
                                   max_features="sqrt",
                                   random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(warm_start=True, max_features='log2',
                                   oob_score=True,
                                   random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(warm_start=True, max_features=None,
                                   oob_score=True,
                                   random_state=RANDOM_STATE))
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 100
    max_estimators = 400

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit([t[2:] for t in target_train], [l[1] for l in target_train])

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    calculate_oob_error()
