#Training classifier (first argument to the script) against a train_path (second argument)

from my_fun import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


train_path = sys.argv[2]

clfRandom = RandomForestClassifier(n_estimators=100)

clfRandom = train_obj(train_path, clfRandom)
#9 stands for maximum compression
joblib.dump(clfRandom, sys.argv[1], 9)
