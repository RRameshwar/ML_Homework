import numpy as np
from collections import Counter

import csv
from graphviz import Digraph


from part1 import *

global dot
global attributes
global leaves

attributes = []
dot = Digraph(comment='Credit Risk')
leaves = 0

def loadTextDataset(txt_file):
    with open(txt_file, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split() for line in stripped if line)
        with open('log.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('title', 'intro'))
            writer.writerows(lines)

    with open("log2.csv") as testCSV:
        readCSV = csv.reader(testCSV, delimiter=',')
        feature_array = []
        global attributes

        for k in range(0, 5):
            feature_array.append([])

        label = []
        next(readCSV)
        row1 = next(readCSV)
        for i in range(1, len(row1)-1):
            attributes.append(row1[i])
        #next(readCSV)
        #print(attributes)

        next(readCSV)
        for row in readCSV:
            if(len(row)>0):
                #print(row)
                label.append(row[-1])
                for i in range(1, len(row)-1):
                    feature_array[i-1].append(row[i])

        X = np.array(feature_array)
        print(X)
        Y = np.asarray(label)
        return X,Y

def drawTree(t,parent,c):
    global attributes
    global dot
    global leaves
    #print(t.isleaf)
    if t.isleaf:
        print("drew leaf")
        name = t.p+str(leaves)
        leaves += 1
        dot.node(name, t.p) #label with most common label
        dot.edge(attributes[parent.i],name,label=c)
    else:
        if parent == "None" or c is None:
            print("drew root")
            dot.node(attributes[t.i], attributes[t.i], color='blue')
        else:
            dot.node(attributes[t.i], attributes[t.i], color='blue')
            print("drew some edges")
            dot.edge(attributes[parent.i], attributes[t.i], label=c)
        for c in t.C:
            #print(t.C)
            drawTree(t.C[c], t, c)



if __name__ == '__main__':
    X,Y = loadTextDataset("credit.txt")
    t1 = Node(X,Y)
    tf = Tree.build_tree(t1)

    if(tf.isleaf):
        dot.node('L', tf.p)
    else:
        drawTree(tf, "None", None)

    dot.render('test-output/round-table.gv', view=True)  # doctest: +SKIP
    'test-output/round-table.gv.pdf'



