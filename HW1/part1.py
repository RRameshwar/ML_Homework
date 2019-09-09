import math
import numpy as np
from collections import Counter

import csv

import sys



# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
	Part 1: Decision Tree (with Discrete Attributes) -- 40 points --
	In this problem, you will implement the decision tree method for classification problems.
	You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''
global recursion_depth
recursion_depth = 0	
#-----------------------------------------------
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

#-----------------------------------------------
class Tree(object):
	'''
		Decision Tree (with discrete attributes). 
		We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
	'''
	#--------------------------
	@staticmethod
	def entropy(Y):
		'''
			Compute the entropy of a list of values.
			Input:
				Y: a list of values, a np array of int/float/string values.
			Output:
				e: the entropy of the list of values, a float scalar
			Hint: you could use collections.Counter.
		'''
		#########################################
		count = Counter()
		for x in Y:
			count[x]+=1

		total = len(Y)
		e = 0
		for x in count:
			P = count[x]/float(total)
			e += P*np.log2(P)

		e = -1*e
		
		#########################################

		return e
	
			
	#--------------------------
	@staticmethod
	def conditional_entropy(Y,X):
		'''
			Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
			Input:
				X: a list of values , a np array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
				Y: a list of values, a np array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
			Output:
				ce: the conditional entropy of y given x, a float scalar
		'''
		#########################################
		ce = 0
		megalist = []
		attrs = []
		total = len(Y)
		print X
		print Y

		for i in range(0, len(X)):
			if(X[i] in attrs):
				megalist[attrs.index(X[i])].append(Y[i])
			else:
				megalist.append([Y[i]])
				attrs.append(X[i])

		print(megalist)
		for j in megalist:
			ce = ce + (len(j)/float(total))*Tree.entropy(j)  #CE is going to be positive, so we haveto subtract from the parent entropy.
 			print ce
		#########################################
		return ce 
	
	
	
	#--------------------------
	@staticmethod
	def information_gain(Y,X):
		'''
			Compute the information gain of y after spliting over attribute x
			InfoGain(Y,X) = H(Y) - H(Y|X) 
			Input:
				X: a list of values, a np array of int/float/string values.
				Y: a list of values, a np array of int/float/string values.
			Output:
				g: the information gain of y after spliting over x, a float scalar
		'''
		#########################################
		parent = Tree.entropy(Y)
		children = Tree.conditional_entropy(Y,X)

		g = parent - children
		
		#print(parent)
		#print(children)
 
		#########################################
		return g


	#--------------------------
	@staticmethod
	def best_attribute(X,Y):
		'''
			Find the best attribute to split the node. 
			Here we use information gain to evaluate the attributes. 
			If there is a tie in the best attributes, select the one with the smallest index.
			Input:
				X: the feature matrix, a np matrix of shape p by n. 
				   Each element can be int/float/string.
				   Here n is the number data instances in the node, p is the number of attributes.
				Y: the class labels, a np array of length n. Each element can be int/float/string.
			Output:
				i: the index of the attribute to split, an integer scalar
		'''
		#########################################
		igs = []
		for i in range(0, len(X)): #Running through each column
			col = X[i]
			igs.append(Tree.information_gain(Y,col)) #The index of an IG corresponds to the column of X it's in
		i = igs.index(max(igs))

		#########################################
		return i

		
	#--------------------------
	@staticmethod
	def split(X,Y,i):
		'''
			Split the node based upon the i-th attribute.
			(1) split the matrix X based upon the values in i-th attribute
			(2) split the labels Y based upon the values in i-th attribute
			(3) build children nodes by assigning a submatrix of X and Y to each node
			(4) build the dictionary to combine each  value in the i-th attribute with a child node.
	
			Input:
				X: the feature matrix, a np matrix of shape p by n.
				   Each element can be int/float/string.
				   Here n is the number data instances in the node, p is the number of attributes.
				Y: the class labels, a np array of length n.
				   Each element can be int/float/string.
				i: the index of the attribute to split, an integer scalar
			Output:
				C: the dictionary of attribute values and children nodes. 
				   Each (key, value) pair represents an attribute value and its corresponding child node.
		'''
		#########################################
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

		sortedX_np = np.array(sortedX)
		print(sortedX_np[0].shape)
		
		for m in range(0, len(sortedX_np)):
			sortedX_np[m] = np.array(sortedX_np[m])

		# for n in range (0, len(sortedX)):
		# 	for m in range (0, n):
		# 		sortedX_np[n][m] = sortedX[n][m]

		# for k in range(0, len(sortedX)):
		# 	for n in range(0, len(sortedX[k])):
		# 		sortedX_np[k][n] = np.array(sortedX[k][n])
		# for k in range(0, len(sortedY)):
		# 	sortedY[k] = np.array(sortedY[k])

		
		#for k in sortedX:
			#np.append(sortedX_np, k)

		#print(sortedX)
		#sortedX_np = sortedX_np.flatten()
		
		print(["THIS IS X!!: ", X])
		print(["THIS IS SORTED X!!: ", sortedX_np])
		print(["THIS IS C!!: ", C])

		sortedY_np = np.asarray(sortedY)


		for j in range(0, len(sortedX_np)):
			print(j)
			n = Node(sortedX_np[j], sortedY_np[j])
			C[sortedX_np[j][i][0]] = n


		return C

	#--------------------------
	@staticmethod
	def stop1(Y):
		'''
			Test condition 1 (stop splitting): whether or not all the instances have the same label. 
	
			Input:
				Y: the class labels, a np array of length n.
				   Each element can be int/float/string.
			Output:
				s: whether or not Conidtion 1 holds, a boolean scalar. 
				True if all labels are the same. Otherwise, false.
		'''
		#########################################
		for x in Y:
			if x == Y[0]:
				s = True			
			else:
				s = False
				break

		
		#########################################
		return s
	
	#--------------------------
	@staticmethod
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
				
	#--------------------------
	@staticmethod
	def most_common(Y):
		'''
			Get the most-common label from the list Y. 
			Input:
				Y: the class labels, a np array of length n.
				   Each element can be int/float/string.
				   Here n is the number data instances in the node.
			Output:
				y: the most common label, a scalar, can be int/float/string.
		'''
		#########################################
		count = Counter()
		for x in Y:
			count[x]+=1

		y = count.most_common(1)[0][0] #Accesses the value of the most common label
 
		#########################################
		return y
		
	#--------------------------
	@staticmethod
	def build_tree(t):
		global recursion_depth
		'''
			Recursively build tree nodes.
			Input:
				t: a node of the decision tree, without the subtree built.
				t.X: the feature matrix, a np float matrix of shape n by p.
				   Each element can be int/float/string.
					Here n is the number data instances, p is the number of attributes.
				t.Y: the class labels of the instances in the node, a np array of length n.
				t.C: the dictionary of attribute values and children nodes. 
				   Each (key, value) pair represents an attribute value and its corresponding child node.
		'''
		#########################################
		recursion_depth+=1
		
		if(Tree.stop1(t.Y)): #Stops if all classes are the same
			recursion_depth-=1
			t.isleaf = True
			t.p = Tree.most_common(t.Y)
			return t #returns final node
		if(Tree.stop2(t.X)): #Stops if all classes are the same
			recursion_depth-=1
			t.isleaf = True
			t.p = Tree.most_common(t.Y)
			return t #returns final node
		else:
			t.p = Tree.most_common(t.Y)
			attr_best = Tree.best_attribute(t.X,t.Y)
			t.i = attr_best
			t.C = Tree.split(t.X ,t.Y, attr_best)
			print(t.C)
			ex = t.C['a'].X
			print(ex)
			print(ex.shape)
			for pair in t.C:				
				return Tree.build_tree(t.C[pair]) #build new tree from node. Will stop when no more building
	
		#########################################
	
	
	#-------------------------
	@staticmethod
	def train(X, Y):
		'''
			Given a training set, train a decision tree. 
			Input:
				X: the feature matrix, a np matrix of shape p by n.
				   Each element can be int/float/string.
				   Here n is the number data instances in the training set, p is the number of attributes.
				Y: the class labels, a np array of length n.
				   Each element can be int/float/string.
			Output:
				t: the root of the tree.
		'''
		#########################################
		t1 = Node(X,Y)
		t = Tree.build_tree(t1)
		#########################################
		return t
	
	@staticmethod
	def printTree(t):
		if Tree.stop1(t.Y): #we're at a leaf!
			print(t.X) 
		else:
			nextNodes = t.C
			for c in t.C:
				return Tree.printTree[t.C[c]] #The next node is the dictionary element corresponding to the value of x at the best attribute
	#--------------------------
	@staticmethod
	def inference(t,x):
		'''
			Given a decision tree and one data instance, infer the label of the instance recursively. 
			Input:
				t: the root of the tree.
				x: the attribute vector, a np vectr of shape p.
				   Each attribute value can be int/float/string.
			Output:
				y: the class labels, a np array of length n.
				   Each element can be int/float/string.
		'''
		#########################################
		if Tree.stop1(t.Y): #we're at a leaf!
			return t.y[0] 
		else:
			nextNode = t.C[x[t.i]] #The next node is the dictionary element corresponding to the value of x at the best attribute
			return Tree.inference(nextNode, x)
 
		#########################################
		#return y

	#--------------------------
	@staticmethod
	def predict(t,X):
		'''
			Given a decision tree and a dataset, predict the labels on the dataset. 
			Input:
				t: the root of the tree.
				X: the feature matrix, a np matrix of shape p by n.
				   Each element can be int/float/string.
				   Here n is the number data instances in the dataset, p is the number of attributes.
			Output:
				Y: the class labels, a np array of length n.
				   Each element can be int/float/string.
		'''
		#########################################
		if t is None:
			print("T is NONE WHYYY")
		Y = []
		for x in X:
			Y.append(Tree.inference(t,x))

		#########################################
		return Y



	#--------------------------
	@staticmethod
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
				X: the feature matrix, a np matrix of shape p by n.
				   Each element can be int/float/string.
				   Here n is the number data instances in the dataset, p is the number of attributes.
				Y: the class labels, a np array of length n.
				   Each element can be int/float/string.
		'''
		#########################################
		with open(filename) as testCSV:
	
			readCSV = csv.reader(testCSV, delimiter=',')
			features = []

			feature_array = []
			for k in range (0,7):
				feature_array.append([])


					
			label = []
			next(readCSV)
			
			for row in readCSV: 
				label.append(row[0])
				for j in range(1,8):
					feature_array[j-1].append(row[j])


			X = np.asarray(feature_array)
			Y = np.asarray(label)

 
		#########################################
		return X,Y

if __name__ == '__main__':
	X, Y = Tree.load_dataset()
	t = Tree.train(X,Y)
	Tree.printTree(t)



