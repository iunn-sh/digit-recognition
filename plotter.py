import recognizer
from os import environ
import plotly
import plotly.plotly as py
from plotly.graph_objs import Scatter, Data

def get_commit_id():
    commit = environ.get('CIRCLE_SHA1')
    print commit[:7]
    return commit[:7]

def get_oob():
    oob = recognizer.random_forest() * 100
    print oob, "%"
    return oob

def plot(commit, oob):
    plotly.tools.set_credentials_file(username=environ.get('PLOTLY_USERNAME'),
                                        api_key=environ.get('PLOTLY_API_KEY'))
    new_data = Scatter(x=commit, y=oob)
    data = Data( [ new_data ] )
    plot_url = py.plot(data, filename='random forest', fileopt='extend')
    print plot_url

if __name__ == "__main__":
    commit = get_commit_id()
    oob = get_oob()
    plot(commit, oob)
