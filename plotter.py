import random_forest
import mnist_softmax
from os import environ
import plotly
import plotly.plotly as py
from plotly.graph_objs import Scatter, Data

def get_commit_id():
    commit = environ.get('CIRCLE_SHA1')
    print commit[:7]
    return commit[:7]

def get_oob_percent():
    forest = random_forest.create_forest()
    oob = random_forest.get_oob(forest) * 100
    print oob, "%"
    return oob

def get_mnist_percent(n_train):
    mnist = mnist_softmax.train_test(n_train) * 100
    print mnist, "%"
    return mnist

def plot(commit, oob):
    plotly.tools.set_credentials_file(username=environ.get('PLOTLY_USERNAME'),
                                        api_key=environ.get('PLOTLY_API_KEY'))
    new_data = Scatter(x=commit, y=oob)
    data = Data( [ new_data ] )
    plot_url = py.plot(data, filename='Random Forest', fileopt='extend')
    print plot_url

if __name__ == "__main__":
    commit = get_commit_id()
    oob = get_oob_percent()
    plot(commit, oob)
    # mnist = get_mnist_percent(6000)
    # plot(commit, mnist)
