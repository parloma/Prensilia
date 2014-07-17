#Testing classifier accurancy

from my_fun import *
from sklearn.externals import joblib
import glob
from os import sep

test_path = 'test_dir'+sep
clf_path = 'clfs'+sep
rf_path = 'RF'+sep

classifiers_paths = glob.glob(rf_path+'*.pkl')

for p in classifiers_paths:
	clf = joblib.load(p)
	print "Running Test - %s" %(p)
	test_obj(test_path, clf)
	print "-----------------------------------------------------------------"
