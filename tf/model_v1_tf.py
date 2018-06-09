import tensorflow as tf 
from tf_utils import get_batch_generator
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


def loadData():
	vocab = utils.glove2dict("../../../../../GitHub/cs224u/vsmdata/glove/glove.6B.50d.txt")  # dict[word] -> numpy array(embed_dim,)
	# self.embedding_matrix: (embed_dim, vocab_size)
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

class RelationClassifier():
	"""Module to convert a sentence of word-vectors into a vector of predicted relation labels.
	"""
	"""
	Args:

	"""
	def __init__(self, emb_matrix, relation2Id, word2Index, trainFileName="SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"):
		# filter size in the 'width' dimension
		self.w = 3
		# number of filters
		self.numFilters = 230
		# stride along the 'width' dimension
		self.stride = 1
		# probability of dropping out a neuron
		self.dropout_rate = 0.5
		# number of epochs
		self.num_epochs = 25
		# number of classes
		self.num_classes = len(relation2Id)
		self.batch_size = 50
		self.trainFileName = trainFileName
		self.word2Index = word2Index
		self.relation2Id = relation2Id
		self.learning_rate = 0.001
		self.numTrainSamples = 8000.0

		# This is the actual operations
		self.add_placeholders()
		self.add_embedding_layer(emb_matrix)
		self.build_graph()
		self.add_loss()
		self.compute_metrics()
		opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.updates = opt.minimize(self.loss)

	def add_placeholders(self):
		self.C1 = tf.placeholder(tf.int32, shape=[None, None])
		self.C2 = tf.placeholder(tf.int32, shape=[None, None])
		self.C3 = tf.placeholder(tf.int32, shape=[None, None])
		self.labels = tf.placeholder(tf.int32, shape=[None,])
		

	def add_embedding_layer(self, emb_matrix):
		embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32)
		self.C1_embed = tf.nn.embedding_lookup(embedding_matrix, self.C1)
		self.C2_embed = tf.nn.embedding_lookup(embedding_matrix, self.C2)
		self.C3_embed = tf.nn.embedding_lookup(embedding_matrix, self.C3)

	def build_graph(self):
		# TODO: Add padding before
		# Use self.C1_embed, self.C2_embed, self.C3_embed
		# self.C1_embed: (batch_size, embed_dim, maxC1Len)
		# Make sure that the dimensions of self.Ci_embed is (b, m, e)!
		# out1: (batch_size, something, numFilters)
		out1 = tf.layers.conv1d(self.C1_embed, self.numFilters, self.w, padding='valid')
		out2 = tf.layers.conv1d(self.C1_embed, self.numFilters, self.w, padding='valid')
		out3 = tf.layers.conv1d(self.C1_embed, self.numFilters, self.w, padding='valid')
		# out1: (batch_size, numFilters)
		out1 = tf.reduce_max(out1, axis=1)
		out2 = tf.reduce_max(out2, axis=1)
		out3 = tf.reduce_max(out3, axis=1)
		# result: (batch_size, 3*numFilters)
		result = tf.concat([out1, out2, out3], 1)
		result = tf.tanh(result)
		result = tf.layers.dropout(result, rate=self.dropout_rate)
		self.logits = tf.contrib.layers.fully_connected(result, 
			self.num_classes, activation_fn=None)

	def add_loss(self):
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=self.logits, labels=self.labels))

	def compute_metrics(self):
		self.predictions = tf.argmax(self.logits, axis=1, output_type=tf.int32)
		self.numEqual = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.labels), tf.float32)) 
 
	def run_train_iter(self, session, batch):
		input_feed = {}
		input_feed[self.C1] = batch[0]
		input_feed[self.C2] = batch[1]
		input_feed[self.C3] = batch[2]
		input_feed[self.labels] = batch[4]
		output_feed = [self.updates, self.loss, self.numEqual]
		_, loss, numEqual = session.run(output_feed, input_feed)
		return loss, numEqual
		
	def train(self, session):
		losses = []	
		accuracies = []
		epoch = 0
		while epoch < self.num_epochs:
			epochLoss = 0
			epochNumEqual = 0
			for batch in get_batch_generator(self.batch_size, self.trainFileName, self.word2Index, self.relation2Id):
				loss, numEqual = self.run_train_iter(session, batch)
				epochLoss += loss
				epochNumEqual += numEqual
			epochLoss /= self.numTrainSamples
			epochNumEqual /= self.numTrainSamples
			losses.appen(epochLoss)
			losses.appen(epochNumEqual)
			epoch += 1
			print("Epoch loss: %d. Accuracy: %d", epochLoss, epochPrecision)

'''
######################################
OLD OPERATIONS

	def convertInputToTensor(self, X):
		"""
		Args:
			X = list[[word1, word2, ...], [word1, word2, ...], ... ] = batch of training examples
			where word1, word2 are each strings indicating the words
		Return:
			X = tensor matrix of dimensions (batch_size, embed_dim, maxSenLen+2w-2)
			sentenceLengths = a list of integers of length batch_size indicating the length of the 
			the sentence for every example
		"""

		# First retrieve the max sentence length
		sentenceLengths = []
		indices = []
		for sentence in X:
			sentenceLengths.append(len(sentence))
			wordsInIndices = []
			for word in sentence:
				if word in self.word2Index:
					wordsInIndices.append(self.word2Index[word])
				# We do not know the word, so add UNK
				else:
					wordsInIndices.append(self.embedding_matrix.shape[1] - 1)
			indices.append(wordsInIndices)
		maxSenLen = max(sentenceLengths)
		toStack = []
		for index in indices:
			s = tf.nn.embedding_lookup(self.embedding_matrix, index)
			# pad left and right with w-1 0's
			s = tf.pad(s, [[0, 0], [self.w-1, self.w-1]])
			# pad right until the max length of the batch
			s = tf.pad(s, [[0, 0], [0, maxSenLen-len(sentence)]])
			# s: (embed_dim, maxSenLen+2w-2)
			toStack.append(s)
		# X:(batch_size, embed_dim, maxSenLen+2w-2)
		X = tf.stack(toStack)
		return X, sentenceLengths

	def calculatePaddingSize(indicesToCut, sentenceLengths):
		maxFirstIndex = max(map(lambda x: x[0], indicesToCut))
		maxSecondIndex = max(map(lambda x: x[1], indicesToCut))


	def findIndicesToCut(indices):
		cutPoints = []
		for index1, index2 in indices:
			point = [(2.0 * index1 + 1 - self.w)/(2*selfstride),
			(2.0 * index2 + 1 - self.w)/(2*selfstride)]
			cutPoints.append(point)
		return cutPoints

	def build_graph2(self):
		X = tf.expand_dims(self.X, axis=-1)
		embed_dim = X.shape[1]
		# out:(batch_size, 1, (convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
		out = tf.layers.conv2d(X, self.num_filters, [embed_dim, self.w], 
			strides = [1, self.stride])
		# out:(batch_size, (convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
		out = tf.squeeze(out)
		batch_size = X.shape[0]
		examples = tf.split(out, num_or_size_splits=batch_size, axis=0)
		vectors = []
		for i in xrange(batch_size):
			# example:(1, (convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
			example = examples[i]
			# example:((convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
			example = tf.squeeze(example)
			A = tf.reduce_max(example[0:indicesToCut[i][0] + 1], axis = 0)
			B = tf.reduce_max(example[indicesToCut[i][0] + 1, 
				indicesToCut[i][1] + 1], axis = 0)
			C = tf.reduce_max(example[indicesToCut[i][1] + 1, :], axis = 0)
			# curResult: (3*self.numFilters,)
			curResult = tf.concat([A, B, C], 0)
			vectors.append(curResult)
			#TODO: check the earlier padding issue here
		# result: (batch_size, 3*self.numFilters)
		result = tf.stack(vectors)
		result = tf.tanh(result)
		result = tf.layers.dropout(result, rate=self.dropout_rate, training=True)
		# SOFTMAX 
		# COMPUTE LOSS
		# FIGURE OUT THE PREDICTION STEP



	def forward_tf(self, X, y, indices):
		"""
		Args:
			X = list[[word1, word2, ...], [word1, word2, ...], ... ] = batch of training examples
			where word1, word2 are each strings indicating the words
			y = list[ int, ... ]               = relation label indices
			indices = list[(index of entity1, index of entity2), (index of entity1, index of entity2), ...] 
					= list of tuples where the first element of the tuple is the index of the first entity
		Return:
			
		"""
		# X:(batch_size, embed_dim, maxSenLen+2w-2)
		X, sentenceLengths = self.convertInputToTensor(X)
		# We will pass X through 2d convolution with in channels
		# set to 1, so expand X's dimensions
		# X:(batch_size, embed_dim, maxSenLen+2w-2, 1)
		X = tf.expand_dims(X, axis=-1)
		embed_dim = X.shape[1]
		# out:(batch_size, 1, (convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
		out = tf.layers.conv2d(X, self.num_filters, [embed_dim, self.w], 
			strides = [1, self.stride])
		# out:(batch_size, (convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
		out = tf.squeeze(out)
		batch_size = X.shape[0]
		examples = tf.split(out, num_or_size_splits=batch_size, axis=0)
		vectors = []
		for i in xrange(batch_size):
			# example:(1, (convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
			example = examples[i]
			# example:((convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
			example = tf.squeeze(example)
			A = tf.reduce_max(example[0:indicesToCut[i][0] + 1], axis = 0)
			B = tf.reduce_max(example[indicesToCut[i][0] + 1, 
				indicesToCut[i][1] + 1], axis = 0)
			C = tf.reduce_max(example[indicesToCut[i][1] + 1, :], axis = 0)
			# curResult: (3*self.numFilters,)
			curResult = tf.concat([A, B, C], 0)
			vectors.append(curResult)
			#TODO: check the earlier padding issue here
		# result: (batch_size, 3*self.numFilters)
		result = tf.stack(vectors)
		result = tf.tanh(result)
		result = tf.layers.dropout(result, rate=self.dropout_rate, training=True)
		# SOFTMAX 
		# COMPUTE LOSS
		# FIGURE OUT THE PREDICTION STEP
		# Continuing as numpy here due to time constraints
		# Tf implementation can be done via padding and splitting
		return out

######################################
OLD TEST

#embedding_matrix, word2Index, index2Word = loadData()
#print(embedding_matrix.shape)
#rc = RelationClassifier(embedding_matrix, word2Index, index2Word)
# i = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], dtype=tf.float32)
# f = tf.constant([[6, 7], [8, 9], [10, 11]], dtype=tf.float32)
# i = tf.expand_dims(i, axis=-1)
# print(i.shape)
# f = tf.expand_dims(f, axis=-1)
# f = tf.expand_dims(f, axis=-1)
# print(f.shape)
# a = tf.nn.conv2d(i, f, [1, 1, 2, 1], 'VALID')
# with tf.Session() as sess:
# 	b=sess.run(a)
# 	print(b.shape)
print("############################")
A = tf.constant([[1, 4], [2, 5], [3, 6], [7, 8]], dtype=tf.float32)
print(A.shape)
A = tf.expand_dims(A, axis=0)
print(A.shape)
F = tf.constant([[9, 10], [11, 12]], dtype=tf.float32)
print(F.shape)
F = tf.expand_dims(F, axis=2)
print(F.shape)
out = tf.nn.conv1d(A, F, 1, 'VALID')
with tf.Session() as sess:
	print(sess.run(out))
'''
	




