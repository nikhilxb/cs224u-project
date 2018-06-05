
import numpy as np
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


def refill_batches(batches, trainDataFile, trainLabelFile):
	print("Refilling batches")



def get_batch_generator():
	trainDataFile = open(trainingDataFilePath)
	trainLabelFile = open(trainingLabelFilePath)
	batches = []
	if len(batches) == 0:
		refill_batches(batches, trainDataFile, trainLabelFile)


def refill_batches_test(lines, f):
	for line in f:
		if line.strip() == "":
			print("IT IS AN EMPTY LINE")
		lines.append(line)
		if len(lines) == 2:
			return


def get_batch_generator_test():
	f = open("testFile.txt")
	lines = []
	while True:
		if len(lines) == 0: # add more batches
			refill_batches_test(lines, f)
		if len(lines) == 0:
			break
		yield lines.pop(0)


def test():
	for number in get_batch_generator_test():
		print(number)

def test1():
	f = open("testFile.txt") 
	for line in f:
		line = line.strip('\n')
		print(line)


if __name__ == "__main__":
	#test()
	test1()