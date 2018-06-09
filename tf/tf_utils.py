import pickle
import tensorflow as tf
import numpy as np
from data_utils import readDictFromFile
import string
import random
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

def loadData(gloveFileName="../../../../../../GitHub/cs224u/vsmdata/glove/glove.6B.50d.txt"):
	vocab = utils.glove2dict(gloveFileName)  # dict[word] -> numpy array(embed_dim,)
	embedding_matrix = np.zeros((len(vocab["and"]), len(vocab)))
	word2Index = {}
	index2Word = {}
	counter = 0
	for word, vec in vocab.items():
		embedding_matrix[:, counter] = vec
		word2Index[word] = counter
		index2Word[counter] = word
		counter += 1
	return embedding_matrix, word2Index, index2Word


def removeEntities(line):
	firstStartIdx = line.find('<e1>')
	firstEndIdx = line.find('</e1>') + len('</e1>')
	firstEntity = line[firstStartIdx:firstEndIdx] 
	line = line.replace(firstEntity, "") 
	secondStartIdx = line.find('<e2>')
	secondEndIdx = line.find('</e2>')
	secondEntity = line[secondStartIdx:secondEndIdx + len('</e2>')]
	line = line.replace(secondEntity, "")
	return line

def findIndices(tokens):
	for i in range(0, len(tokens)):
		token = tokens[i]
		if "<e1>" in token:
			beginFirstEntitiy = i
		if "</e1>" in token:
			endFirstEntity = i
		if "<e2>" in token:
			beginSecondEntity = i
		if "</e2>" in token:
			endSecondEntity = i
	return beginFirstEntitiy, endFirstEntity, beginSecondEntity, endSecondEntity

def stripMore(c):
	newC = []
	d = {}
	keys = ["/", "<", ">"]
	for key in keys:
		d[key] = None
	table = str.maketrans(d)
	for token in c:
		newC.append(token.translate(table))
	return newC

def convertToNums(c, word2Index):
	newC = []
	for word in c:
		if word in word2Index:
			newC.append(word2Index[word])
		else:
			newC.append(word2Index["<UNK>"])
	return newC

'''
Given a line in the form:
id 	"word1 word2 ... <e1>entity1_1 entity1_2 ... entity1_n</e1> ... <e2>entity2_1 ... entity2_m</e2> ... wordk."
Convert the line to
([word1_idx, word2_idx, ...], [...], [...], id)
where the inside lists are actually numpy arrays
'''
def convertLineToIdx(line, table, word2Index):
	# Remove all punctuations
	line = line.translate(table)
	# Make lowercase
	line = line.lower()
	tokens = line.split()
	# Get sentence id
	sentenceId = int(tokens[0])
	tokens = tokens[1:]
	beginFEntity, endFEntity, beginSEntity, endSEntity = findIndices(tokens)
	c1 = tokens[:beginFEntity]
	c2 = tokens[endFEntity + 1:beginSEntity]
	c3 = tokens[endSEntity + 1:]
	c1 = stripMore(c1)
	c2 = stripMore(c2)
	c3 = stripMore(c3)
	c1 = convertToNums(c1, word2Index)
	c2 = convertToNums(c2, word2Index)
	c3 = convertToNums(c3, word2Index)
	return (c1, c2, c3, sentenceId)

def getRelationId(line, relation2Id):
	idx = line.find("(")
	if idx == -1:
		return int(relation2Id[line.split()[0]])
	else:
		return int(relation2Id[line[:idx]])

def findLongestSeq(cs):
	maxLen = None
	for c in cs:
		if maxLen is None or len(c) > maxLen:
			maxLen = len(c)
	return maxLen


def refill_batches(batches, trainDataFile, batch_size, word2Index, relation2Id, lineCounter):
	#print("Refilling batches")
	# an example will be in the form [c1, c2, c3, label] where ci is a list, label is an integer
	examples = []
	# Will be used to strip elements from a line
	table = str.maketrans({key: None for key in string.punctuation if key != "-" and key != "<" and key != ">" and key != "/"})
	for line in trainDataFile:
		if lineCounter % 4 == 2 or lineCounter % 4 == 3:
			lineCounter += 1
			continue
		if lineCounter % 4 == 0:
			example = convertLineToIdx(line, table, word2Index)
			lineCounter += 1
			continue
		elif lineCounter % 4 == 1:
			lineCounter += 1
			relationId = getRelationId(line, relation2Id)
			example = (example[0], example[1], example[2], example[3], relationId)
		examples.append(example)
		if len(examples) == batch_size * 5:
			break
	random.shuffle(examples)
	# Divide examples into batches
	for i in range(0, len(examples), batch_size):
		curBatch = examples[i:i+batch_size]
		# Pad batches
		# curBatch: [(c1_1, c1_2, ...), (c2_1, c2_2, ...), (c3_1, c3_2, ...), (label1, label2, ...)]
		curBatch = list(zip(*curBatch))
		for i in range(0, len(curBatch)):
			if i >= 3:
				break
			maxLen = findLongestSeq(curBatch[i])
			cs = []
			# Convert all elements to numpy and pad
			for j in range(0, len(curBatch[i])):
				c = curBatch[i][j]
				c = np.asarray(c, dtype=np.int32)
				c = np.pad(c, (0, maxLen-len(c)), 'constant', constant_values=int(word2Index["<PAD>"]))
				cs.append(c)
			curBatch[i] = np.asarray(cs)
		batches.append(curBatch)
	return lineCounter
	#print("Refilling batches done")

# Every batch will be in the form of [[C1, C2, C3, (label1, label2, label3, ..., label_batch_size)]
# where C1: (batch_size, C1_len)
# where C2: (batch_size, C2_len)
# where C3: (batch_size, C3_len)
# Ci_len is not known beforehand
def get_batch_generator(batch_size, trainDataFileName, word2Index, relation2Id):
	trainDataFile = open(trainDataFileName)
	batches = []
	lineCounter = 0
	while True:
		if len(batches) == 0:
			lineCounter = refill_batches(batches, trainDataFile, batch_size, word2Index, relation2Id, lineCounter)
		if len(batches) == 0:
			return
		yield batches.pop(0)


def get_batch_generator_test():
	word2Index = {}
	readDictFromFile(word2Index, "SemEval2010_task8_all_data/cleaned/cleaned_word2Index.txt")
	relation2Id = {}
	readDictFromFile(relation2Id, "SemEval2010_task8_all_data/cleaned/cleaned_relation2Id.txt")
	for batch in get_batch_generator(10, "SemEval2010_task8_all_data/SemEval2010_task8_training/small_TRAIN_FILE.TXT", word2Index, relation2Id):
		print(batch[4])

def convertcToString(matrix, index2Word):
	newMatrix = []
	for arr in matrix:
		c = []
		for num in arr:
			c.append(index2Word[str(num)])
		newMatrix.append(c)
	return newMatrix

def testConv():
	A = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=tf.float32)
	A = tf.expand_dims(A, axis=0)
	A = tf.transpose(A, perm=[0, 2, 1] )
	print(A.shape)
	B = tf.constant([[9, 11], [10, 12]], dtype=tf.float32)
	B = tf.expand_dims(B, axis=2)
	print(B.shape)
	C = tf.nn.conv1d(A, B, 1, 'VALID')
	with tf.Session() as sess:
		print(A.eval())
		print(sess.run(C).shape)

def testMax():
	A = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=tf.float32)
	A = tf.expand_dims(A, axis=0)
	A = tf.transpose(A, perm=[0, 2, 1])
	print(A.shape)
	B = tf.reduce_max(A, axis=1)
	with tf.Session() as sess:
		print(sess.run(B).shape)


def test():
	index2Word = {}
	readDictFromFile(index2Word, "SemEval2010_task8_all_data/cleaned/cleaned_index2Word.txt")
	for batch in get_batch_generator(10, "SemEval2010_task8_all_data/cleaned/small_cleaned_train.txt"):
		pass
		#print(batch[0])
		#print(convertcToString(batch[0], index2Word))
		#print(batch)

def test1():
	f = open("testFile.txt") 
	for line in f:
		line = line.strip('\n')
		print(line)

if __name__ == "__main__":
	get_batch_generator_test()
	#test1()
