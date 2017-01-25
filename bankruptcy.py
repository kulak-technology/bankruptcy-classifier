# Author: Kacper "kulak" Kula

from scipy.io import arff
from cStringIO import StringIO
from sklearn import tree
import math
from sklearn.preprocessing import Imputer,PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
import copy

TESTSET_RATIO = 0.05


def convertToFormat (data):
    d = []
    l = []
    for row in data:
        d.append(row[0:len(row)-1])
        l.append(int(row[len(row)-1]))
    return (d, l)

def separateData (data, test_count):
    data = data.tolist()
    train_from = int(math.floor(test_count/2))
    train_to = int(len(data)-math.ceil(test_count/2))
    test_data = data[0:train_from] + data[train_to:]
    train_data = data[train_from:train_to]
    return (test_data, train_data)

def addPolyFeatures (fit, data):
    poly = PolynomialFeatures(2)
    poly.fit(fit)
    return poly.transform(data)

def runClassifier (idx, filename):
    print '---------'
    print 'Running classifier on ', filename
    file_data=''
    with open(filename) as f:
        file_data = f.read()

    f = StringIO(file_data)
    data,meta = arff.loadarff(f)

    test_count = int(TESTSET_RATIO * len(data))

    test_data_combined, train_data_combined = separateData(data, test_count)

    test_data, test_labels = convertToFormat(test_data_combined)
    train_data, train_labels = convertToFormat(train_data_combined)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    sum_data = test_data + train_data
    imp.fit(sum_data)
    test_data = imp.transform(test_data).tolist()
    train_data = imp.transform(train_data).tolist()

    sum_data = copy.deepcopy(test_data + train_data)

    test_data = addPolyFeatures(sum_data, test_data)
    train_data = addPolyFeatures(sum_data, train_data)

    clf = AdaBoostClassifier()
    clf.fit(train_data, train_labels)
    print 'Ada Boost:', clf.score(test_data, test_labels)
    joblib.dump(clf, 'model_dumps/model' + str(idx) + '.pkl')

for i in [1,2,3,4,5]:
    runClassifier(i, 'Dane/' + str(i) + 'year.arff')
