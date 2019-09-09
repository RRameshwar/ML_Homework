import numpy
import csv
from collections import Counter


class Node:
	'''
		Decision Tree Node (with discrete attributes)
		Inputs: 
			X: the data instances in the node, a np matrix of shape p by n.
			   Each element can be int/float/string.
			   Here n is the number data instances in the node, p is the number of attributes.
			Y: the class labels, a np array of length n.
			   Each element can be int/float/string.
			i: the index of the attribute being tested in the node, an integer scalar 
			C: the dictionary of attribute values and children nodes. 
			   Each (key, value) pair represents an attribute value and its corresponding child node.
			isleaf: whether or not this node is a leaf node, a boolean scalar
			p: the label to be predicted on the node (i.e., most common label in the node).
	'''
	def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
		self.X = X
		self.Y = Y
		self.i = i
		self.C= C
		self.isleaf = isleaf
		self.p = p

def load_dataset(filename = 'data1.csv'):
	'''
	Load dataset 1 from the CSV file: 'data1.csv'. 
	The first row of the file is the header (including the names of the attributes)
	In the remaining rows, each row represents one data instance.
	The first column of the file is the label to be predicted.
	In remaining columns, each column represents an attribute.
	Input:
	filename: the filename of the dataset, a string.
	Output:
	X: the feature matrix, a numpy matrix of shape p by n.
	Each element can be int/float/string.
	Here n is the number data instances in the dataset, p is the number of attributes.
	Y: the class labels, a numpy array of length n.
	Each element can be int/float/string.
	'''
	#########################################
	with open(filename, newline='') as testCSV:
	
		readCSV = csv.reader(testCSV, delimiter=',')
		features = []
		feature_array = []
		label = []
		next(readCSV)
		for row in readCSV:
			#print(row) 
			label.append(row[0])
			for j in range(1,8):
				#print(row[j])
				features.append(row[j])
			feature_array.append(features)
			features = []

	X = numpy.array(label)
	Y = numpy.array(feature_array)

	return X, Y


def entropy(Y):
	count = Counter()
	for x in Y:
		count[x]+=1
	total = len(Y)
	
	
	e = 0
	for x in count:
		P = count[x]/total
		e += P*numpy.log2(P)

	e = -1*e

	#########################################


	#########################################
	return e


def conditional_entropy(Y,X):
	'''
		Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
		Input:
			X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
			Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
		Output:
			ce: the conditional entropy of y given x, a float scalar
	'''
	#########################################
	ce = 0
	megalist = []
	attrs = []
	total = len(Y)

	for i in range(0, len(X)):
		if(X[i] in attrs):
			megalist[attrs.index(X[i])].append(Y[i])
		else:
			megalist.append([])
			attrs.append(X[i])
			megalist[attrs.index(X[i])].append(Y[i])

	print(megalist)
	for j in megalist:
		print(j)
		ce = ce + (len(j)/total)*entropy(j)  #CE is going to be positive, so we haveto subtract from the parent entropy.
		print(ce)

	#########################################
	return ce 

def split(X,Y,i):
	sortedX = []
	sortedY = []
	megalist = []
	attrs = []
	C = {}

	num_attrs = len(X)

	attr_responses = X[i]

	num_samples = len(attr_responses)

	for j in range(0, num_samples):
		att = attr_responses[j]
		if att in attrs:
			att_ind = attrs.index(att)
			for n in range(0, len(X)):
				sortedX[att_ind][n].append(X[n][j])
			sortedY[att_ind].append(Y[j])

		else:
			sortedX.append([])
			sortedY.append([])
			
			for n in range(0, num_attrs):
				sortedX[-1].append([X[n][j]])
			sortedY[-1].append(Y[j])
			attrs.append(att)

	for j in range(0, len(sortedX)):
		print(["sortedX: ", sortedX[j]])
		n = Node(sortedX[j], sortedY[j])
		C[sortedX[j][i][0]] = n

	print(["Sorted Y: ", sortedY])

 
	#for j in range(0, len(attrs)):
	#	n = Node(sortedX[j], sortedY[j])
	#	C[sortedX[j][0]] = n

	#########################################
	return C

def stop2(X):
	'''
		Test condition 2 (stop splitting): whether or not all the instances have the same attribute values. 
		Input:
			X: the feature matrix, a np matrix of shape p by n.
			   Each element can be int/float/string.
			   Here n is the number data instances in the node, p is the number of attributes.
		Output:
			s: whether or not Conidtion 2 holds, a boolean scalar. 
	'''
	#########################################
	for i in X:
		for n in i:
			if (n==i[0]):
				s = True
			else:
				s = False
				break

	#########################################
	return s

if __name__ == '__main__':
	X = numpy.array([['apple','apple','apple','apple'],
                  ['high','high','high','high'],
                  ['a','a','a','a']])
	Y = numpy.array(['good','bad','okay','perfect'])
	print(stop2(X))