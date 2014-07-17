#Validation script - Cross Correlation or Simple One-Way Testing

import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from my_fun import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


train_path = 'datasetASL/ds9/'
train_path = 'datasetECCV/train/'
test_path = 'datasetECCV/test/'

clfRandom = RandomForestClassifier(n_estimators=100)
clfRandomAdd = joblib.load('tree_Random_class_4.pkl')


clf = clfRandom

#validation = test_obj(train_path, clf)
#validation = cross_test_obj(train_path, clf)

validation = conf_test_obj(train_path, test_path, clf)
