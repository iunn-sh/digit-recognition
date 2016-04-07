import recognizer
from os import environ
import plotly.plotly as py
from plotly.graph_objs import *

def get_commit_id():
    commit = environ.get('CIRCLE_SHA1')
    print commit
    return commit

def get_oob():
    oob = recognizer.random_forest() * 100
    print oob, "%"
    return oob

def plot(commit, oob):
    username = environ.get('PLOTLY_USERNAME')
    api_key = environ.get('PLOTLY_API_KEY')
    py.sign_in(username, api_key)
    new_data = Scatter(x=[commit], y=[oob] )
    data = Data( [ new_data ] )
    plot_url = py.plot(data, filename='oob_accuracy', fileopt='extend')

if __name__ == "__main__":
    commit = get_commit_id()
    oob = get_oob()
    # plot(commit, oob)
