from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from numpy import genfromtxt, savetxt, std, argsort, hstack
from os import path
import csv

NUM_TREE = 250

def create_forest():
    # create the training & test sets
    script_dir = path.dirname(__file__)  # <-- absolute dir the script is in

    label = read_csv_to_list(path.join(script_dir, 'data/label.csv'))
    train = read_csv_to_list(path.join(script_dir, 'feature/deskew_data_train.csv'))
    test = read_csv_to_list(path.join(script_dir, 'feature/deskew_data_test-new.csv'))
    label_train = link_by_key(label, train)

    # include xy_sum in feature set
    train_xy = read_csv_to_list(path.join(script_dir, 'feature/xy_sum_data_train.csv'))
    label_train = link_by_key(label_train, train_xy)
    test_xy = read_csv_to_list(path.join(script_dir, 'feature/xy_sum_data_test-new.csv'))
    test = link_by_key(test, test_xy)

    # include bbox in feature set
    train_bbox = read_csv_to_list(path.join(script_dir, 'feature/bbox_data_train.csv'))
    label_train = link_by_key(label_train, train_bbox)
    test_bbox = read_csv_to_list(path.join(script_dir, 'feature/bbox_data_test-new.csv'))
    test = link_by_key(test, test_bbox)

    print "training set count =", len(label_train)
    print "testing set count =", len(test)

    # create and train the random forest
    forest = RandomForestClassifier(n_estimators=NUM_TREE, oob_score=True, n_jobs=-1)
    forest.fit([t[2:] for t in label_train], [l[1] for l in label_train])
    prob = forest.predict_proba([t[1:] for t in test])
    print prob
    submit = hstack([[[t[0]] for t in test], prob])
    # for row in submit:
    #     h = False
    #     for i in range(1,len(row)):
    #         if(float(row[i]) >= 0.4):
    #             h = True
    #             #print 'row=%s,p=%s > 0.9' %(row[0],row[i])
    #             break
    #     if(not h):
    #         print row[0]

    # combine sample name & predicted probability
    savetxt(path.join(script_dir, 'data/submit.csv'),
            submit, delimiter=',', fmt='%s')

    return forest

def read_csv_to_list(path):
    l = []
    with open(path, 'rb') as f:
        reader = csv.reader(f)
        l = list(reader)
    return l

def link_by_key(a, b):
    # use dictionary for better performance
    a_dict = { i[0]:i[1:] for i in a }
    b_dict = { j[0]:j[1:] for j in b }
    print "length of arrays =", len(a_dict), len(b_dict)

    ab = []
    for key in a_dict:
        line = []
        line.append(key)
        line.extend(a_dict[key])
        line.extend(b_dict.get(key))
        ab.append(line)

    return ab

def get_oob(forest):
    print "out-of-bag accuracy =", forest.oob_score_
    return forest.oob_score_


if __name__ == "__main__":
    forest = create_forest()
    get_oob(forest)
