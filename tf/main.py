import tensorflow as tf
from tf_utils import loadData
import random
from data_utils import readDictFromFile
import numpy as np
from model_v1_tf import RelationClassifier
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

def createEmbedMatrix(embeddingFileName="../../../../../../GitHub/cs224u/vsmdata/glove/glove.6B.50d.txt"):
	vocab = utils.glove2dict(embeddingFileName)
	emb_matrix = np.zeros((len(vocab) + 2, vocab["the"].shape[0]))
	word2Index = {}
	index2Word = {}
	index = 0
	for word in vocab:
		emb_matrix[index] = vocab[word]
		word2Index[word] = index
		index2Word[index] = word
		index += 1
	# Add UNK
	word2Index["<UNK>"] = len(word2Index)
	index2Word[word2Index["<UNK>"]] = "<UNK>"
	unkValue = []
	# pick unkValue as random values
	for i in range(0, vocab["the"].size):
		randomKey = random.choice(list(vocab.keys()))
		unkValue.append(vocab[randomKey][i])  
	emb_matrix[word2Index["<UNK>"]] = np.asarray(unkValue)
	# Add PAD
	word2Index["<PAD>"] = len(word2Index)
	index2Word[word2Index["<PAD>"]] = "<PAD>"
	emb_matrix[word2Index["<PAD>"]] = np.zeros_like(vocab["the"])
	return emb_matrix, word2Index, index2Word

def main(unused_argv):

	emb_matrix, word2Index, index2Word = createEmbedMatrix()
	relation2Id = {} 
	# This is a required document
	readDictFromFile(relation2Id, "SemEval2010_task8_all_data/cleaned/cleaned_relation2Id.txt")

	# Initialize model
	print("Initializing model")
	model = RelationClassifier(emb_matrix, relation2Id, word2Index)
	print("Initializing model finished")

	with tf.Session() as sess:
		# Initialize variables
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())
		model.train(sess)
		#print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())		


def test():
	a = tf.constant([1, 2, 3, 4, 5])
	b = tf.constant([1, 2, 0, 4, 1])
	c = tf.reduce_sum(tf.cast(tf.equal(a, b), tf.float32))/tf.shape(a)
	with tf.Session() as sess:
		print(sess.run(c))
		#print()

if __name__ == "__main__":
    tf.app.run()
    #test()