# digit-recognition
[![Circle CI](https://circleci.com/gh/jbytw/digit-recognition.svg?style=shield)](https://circleci.com/gh/jbytw/digit-recognition)

## Commit History
[![out-of-bag accuracy](https://plot.ly/~jbytw/16/random-forest.png)](https://plot.ly/~jbytw/16/random-forest)

## Development Setup
* python http://apple.stackexchange.com/questions/209572/how-to-use-pip-after-the-el-capitan-max-os-x-upgrade
* scikit-learn http://scikit-learn.org/stable/install.html

## Folder Structure
* `data`    : raw data for training / testing
* `ref`     : collected sample code from the internet
* `feature` : extracted feature set from `data`

## File Usage
* `extractor.py`  : pre-process, only run 1 time for a dataset
* `recognizer.py` : train, test, create .csv for submission (main function)
* `optimizer.py`  : calculate then plot out-of-bag accuracy for the range of trees (n_estimators)
* `plotter.py`    : plot history accuracy to plotly, run at CircleCI

## Feature Extraction
* color difference / slope
* histogram of oriented gradients
