import sys
import numpy as np
import random_forest

# python finalizer.py submit/submit_6_-0.069444.csv submit/submit_8.csv

def gen_finalize_ans(fn, output_fn):
    prob_before = random_forest.read_csv_to_list(fn)
    prob_after = []
    max_prob = []

    for row in prob_before:
        mx = max(np.array(row[1:], dtype=np.float))
        max_prob.append(mx)
    avg_prob = np.average(max_prob)
    print "average max(prob) =", avg_prob

    threshold = 0.9
    # threshold = avg_prob
    print "finalize threshold =", threshold

    not_sure_prob = []
    for row in prob_before:
        name = row[0]
        prob = np.array(row[1:], dtype=np.float)
        if max(prob) < threshold:
            not_sure_prob.append(max(prob))
            linked = [name] + ['{}'.format(x) for x in prob]
        else:
            new_prob = np.zeros(len(prob));
            new_prob[np.argmax(prob)] = 1
            # print prob
            # print new_prob
            linked = [name] + ['{}'.format(x) for x in new_prob]
        # print linked
        prob_after.append(linked)
    print "max(prob) < threshold count =", len(not_sure_prob)
    print "total count =", len(prob_after)

    with open(output_fn, 'wb') as f:
        for l in prob_after:
            f.write(','.join(l)+'\n')


if __name__ == '__main__':
    fn = sys.argv[1]
    output_fn = sys.argv[2]
    gen_finalize_ans(fn, output_fn)
