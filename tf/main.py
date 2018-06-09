import tensorflow as tf
from tf_utils import loadData
import random
from data_utils import readDictFromFile
import numpy as np
from model_v1_tf import RelationClassifier
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

def createEmbedMatrix(embeddingFileName="../../../glove.6B.50d.txt"):
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
	hyperparams={"numFilters": 230, "dropout_rate": 0.5, "num_epochs": 75, "learning_rate": 0.001, 
	"reg_constant": 0.05}
	
	emb_matrix, word2Index, index2Word = createEmbedMatrix()
	relation2Id = {} 
	# This is a required document
	readDictFromFile(relation2Id, "SemEval2010_task8_all_data/cleaned/cleaned_relation2Id.txt")
	dropout_rateL = [0, 0.5]
	reg_constantL = [0, 0.005]
	for dropout_rate in dropout_rateL:
		for reg_constant in reg_constantL:
			hyperparams["dropout_rate"] = dropout_rate
			hyperparams["reg_constant"] = reg_constant
			# Initialize model
			print("Initializing model")
			model = RelationClassifier(emb_matrix, relation2Id, word2Index, hyperparams, experimentName="TestReg")
			print("Initializing model finished")
			with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
				# Initialize variables
				sess.run(tf.local_variables_initializer())
				sess.run(tf.global_variables_initializer())
				model.train(sess)
				print("Training done")

if __name__ == "__main__":
    tf.app.run()
    #test()
