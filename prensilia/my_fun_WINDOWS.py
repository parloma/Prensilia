#Collection of fundamental functions

import xml.etree.ElementTree as ET
#import numpy
import glob
from collections import Counter
from sklearn.cross_validation import *

#Our novel second classification layer
def joints2dist(joints):
	dist = []
	for i in range(0,len(joints)):
		for j in range(i+1,len(joints)):
			dist.append(numpy.linalg.norm(joints[i]-joints[j]))
	return dist

def compute_dists(file_path):
	tree = ET.parse(file_path)
	root = tree.getroot()
	t = []
	for child in root:
		t.append(numpy.array([float(x) for x in child.find('coords').text.split()]))

	dist = []
	for i in range(0,len(t)):
		for j in range(i+1,len(t)):
			dist.append(numpy.linalg.norm(t[i]-t[j]))
	
	return dist

#Automatically create patterns and label to parse input files
def create_patterns(paths):
	patt = []
	for p in paths:
		patt.append(compute_dists(p))
	return patt
	
def create_labels(paths):
	Y = []
	for p in paths:
		Y.append(p.split('/')[3])
	return Y

#Training against patterns (input data) and a priori labels (Y)
def train_obj(path_training_set, classifier_obj):
	paths = glob.glob(path_training_set+'*/*/*.xml')
	X = create_patterns(paths)
	Y = create_labels(paths)

	print Counter(Y)

	classifier_obj = classifier_obj.fit(X,Y)
	return classifier_obj

#Testing against patterns (input data) and a priori labels (Y)
def test_obj(path_test, classifier_obj):
	paths = glob.glob(path_test+'*/*/*.png')
	X = create_patterns(paths)
	Y = create_labels(paths)

	print "Debug Paths"

	cY = Counter(Y)
	classified = classifier_obj.predict(X).tolist()
	cC = Counter(classified)

	print "Debug Classified"

	result = {'total': {'correct':0, 'tot':len(Y)}}
	for world in cY:
		result[world] = {'correct':0, 'tot':cY[world], 'found':cC[world]}
	

	for i in range(0, len(classified)):
		if classified[i] == Y[i]:
			result['total']['correct']+=1
			result[Y[i]]['correct']+=1
			
	for key in result.keys():
		if key!='total':
			print "%10s\t\t%d/%d (%d)\t\t%0.2f%s" % (key, result[key]['correct'], result[key]['tot'], result[key]['found'], float(result[key]['correct'])/result[key]['tot']*100,'%')

	key = 'total'		
	print "\n%10s\t\t%d/%d\t\t\t%0.2f%s\n\n" % (key, result[key]['correct'], result[key]['tot'], float(result[key]['correct'])/result[key]['tot']*100, '%')


def conf_test_obj(path_train, path_test, classifier_obj):
	paths = glob.glob(path_train+'*/*/*.xml')
	X = create_patterns(paths)
	Y = create_labels(paths)
	paths = glob.glob(path_test+'*/*/*.xml')
	Z = create_patterns(paths)
	W = create_labels(paths)

	print "OK Paths"

	cY = Counter(Y)
	classified = classifier_obj.fit(X,Y).predict(Z).tolist()
	cC = Counter(classified)

	print "OK Classified"

	result = {'total': {'correct':0, 'tot':len(Y)}}
	matrix= {'total': {'A':0,'B':0,'C':0,'D':0,'E':0,'F':0,'H':0,'I':0,'K':0,'L':0,'M':0,'N':0,'O':0,'P1':0,'P2':0,'Q':0,'R':0,'S1':0,'S2':0,'T':0,'U':0,'V':0,'W':0,'X':0,'Y':0}}
	for word in cY:
		result[word] = {'correct':0, 'tot':cY[word], 'found':cC[word]}
		matrix[word] = {'A':0,'B':0,'C':0,'D':0,'E':0,'F':0,'H':0,'I':0,'K':0,'L':0,'M':0,'N':0,'O':0,'P1':0,'P2':0,'Q':0,'R':0,'S1':0,'S2':0,'T':0,'U':0,'V':0,'W':0,'X':0,'Y':0}


	for i in range(0, len(classified)):
		if classified[i] == W[i]:
			result['total']['correct']+=1
			result[W[i]]['correct']+=1

		matrix[W[i]][classified[i]]+=1
			
	for key in result.keys():
		if key!='total':
			print "%10s\t\t%d/%d (%d)\t\t%0.2f%s" % (key, result[key]['correct'], result[key]['tot'], result[key]['found'], float(result[key]['correct'])/result[key]['tot']*100,'%')

	key = 'total'		
	print "\n%10s\t\t%d/%d\t\t\t%0.2f%s\n\n" % (key, result[key]['correct'], result[key]['tot'], float(result[key]['correct'])/result[key]['tot']*100, '%')

	#print matrix
	
	sign=['A','B','C','D','E','F','H','I','K','L','M','N','O','P1','P2','Q','R','S1','S2','T','U','V','W','X','Y']

	f = open("fileNEW.txt", "w")
	for i in sign:
		for j in sign:
			print >>f, ' %d' %matrix[i][j]
	f.close()

#Cross validation 
def cross_test_obj(path_test, classifier_obj):#classifier NOT already fitted!
	paths = glob.glob(path_test+'*/*/*.xml')
	X = numpy.asarray(create_patterns(paths))
	Y = numpy.asarray(create_labels(paths))

	#print X
	#print Y

	#print "Debug Paths"

	#Different paths of cross validation from sklearn library
	#cv = StratifiedKFold(y, k=6)
	cv = LeavePOut(len(Y), 500)
	#cv = LeaveOneOut(len(Y), indices=True)

	#print "Debug cv"
	#print cv

	for i, (train, test) in enumerate(cv):
		#print "Debug Train"
		#print train
		#print "Debug Test"
		#print test

		if i < 10:

			cY = Counter(Y)

			#for j in enumerate(train):
				#classifiedTMP = classifier_obj.fit(X[j], Y[j])

			#for j in enumerate(test):
				##classified = classifiedTMP.predict(X[j]).tolist()
				#classified = classifiedTMP.predict(X[j]).tolist()

			classified = classifier_obj.fit(X[train], Y[train]).predict(X[test]).tolist()
			#print classified

			cC = Counter(classified)

			result = {'total': {'correct':0, 'tot':len(Y)}}
			for world in cY:
				result[world] = {'correct':0, 'tot':cY[world], 'found':cC[world]}


			for i in range(0, len(classified)):
				if classified[i] == Y[i]:
					result['total']['correct']+=1
					result[Y[i]]['correct']+=1

			partial = 0

			for key in result.keys():
				if key!='total':
					print "%10s\t\t%d/%d (%d)\t\t%0.2f%s" % (key, result[key]['correct'], result[key]['tot'], result[key]['found'], float(result[key]['correct'])/(0.001+result[key]['found'])*100,'%')
					partial = partial + result[key]['found']

			if partial == 0:
				partial = 0.001

			key = 'total'		
			#print "\n%10s\t\t%d/%d\t\t\t%0.2f%s\n\n" % (key, result[key]['correct'], result[key]['tot'], float(result[key]['correct'])/result[key]['tot']*100, '%')
			print "\n%10s\t\t%d/%d\t\t\t%0.2f%s\n\n" % (key, result[key]['correct'], partial, float(result[key]['correct'])/partial*100, '%')

#Support functions for cross validation
def create_training_set_cross(path_test):
	paths = glob.glob(path_test+'*/*/*.xml')
	X = numpy.asarray(create_patterns(paths))
	return X

def create_test_set_cross(path_test):
	paths = glob.glob(path_test+'*/*/*.xml')
	Y = numpy.asarray(create_labels(paths))
	return Y

def cross_test_objXY(path_test, classifier_obj, X, Y):#classifier NOT already fitted!

	#print X
	#print Y

	#print "Debug Paths"

	#cv = StratifiedKFold(y, k=6)
	cv = LeavePOut(len(Y), 500)
	#cv = LeaveOneOut(len(Y), indices=True)

	#print "Debug cv"
	#print cv

	for i, (train, test) in enumerate(cv):
		#print "Debug Train"
		#print train
		#print "Debug Test"
		#print test

		if i < 5:
			cY = Counter(Y)

			#for j in enumerate(train):
				#classifiedTMP = classifier_obj.fit(X[j], Y[j])

			#for j in enumerate(test):
				##classified = classifiedTMP.predict(X[j]).tolist()
				#classified = classifiedTMP.predict(X[j]).tolist()

			classified = classifier_obj.fit(X[train], Y[train]).predict(X[test]).tolist()
			#print classified

			cC = Counter(classified)

			result = {'total': {'correct':0, 'tot':len(Y)}}
			for world in cY:
				result[world] = {'correct':0, 'tot':cY[world], 'found':cC[world]}


			for i in range(0, len(classified)):
				if classified[i] == Y[i]:
					result['total']['correct']+=1
					result[Y[i]]['correct']+=1

			partial = 0

			for key in result.keys():
				if key!='total':
					print "%10s\t\t%d/%d (%d)\t\t%0.2f%s" % (key, result[key]['correct'], result[key]['tot'], result[key]['found'], float(result[key]['correct'])/(0.001+result[key]['found'])*100,'%')
					partial = partial + result[key]['found']

			if partial == 0:
				partial = 0.001

			key = 'total'		
			#print "\n%10s\t\t%d/%d\t\t\t%0.2f%s\n\n" % (key, result[key]['correct'], result[key]['tot'], float(result[key]['correct'])/result[key]['tot']*100, '%')
			print "\n%10s\t\t%d/%d\t\t\t%0.2f%s\n\n" % (key, result[key]['correct'], partial, float(result[key]['correct'])/partial*100, '%')
		else:
			return
