import tensorflow as tf 
import numpy as np


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
		embedding_matrix = numpy array of dimensions (embed_dim, vocab_size)
		word2Index = dictionary of size vocab_size, mapping string to integer
		index2Word = dictionary of size vocab_size, mapping integer to string
		NOTE THAT WE ADD THE UNK_TOKEN HERE, AND VOCAB_SIZE INCREASES BY 1, BUT
		WE WILL STILL CALL vocab_size AND NOT vocab_size + 1 FOR CONVENIENCE
	"""
	def __init__(self, embedding_matrix, word2Index, index2Word):
		# add unk token to embedding matrix
		np.random.seed(1)
		# create a random vector
		# unk_vector: 
		unk_vector = np.random.randint(-1, high=1, size=(embedding_matrix.shape[0], 1))
		# add the unk_vector to embedding_matrix
		embedding_matrix = np.append(embedding_matrix, unk_vector, axis = 1)
		self.embedding_matrix = tf.convert_to_tensor(embedding_matrix)
		self.word2Index = word2Index
		self.index2Word = index2Word
		# filter size in the 'width' dimension
		self.w = 3
		# number of filters
		self.numFilters = 5
		# stride along the 'width' dimension
		self.stride = 1
		# probability of dropping out a neuron
		self.dropout_rate = 0.2
		
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

	def forward_np(self, X, y, indices):
		"""
		Args:
			X = np matrix of size (batch_size, (convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
			y = list[ int, ... ]               = relation label indices
			indices = list[(index of entity1, index of entity2), (index of entity1, index of entity2), ...] 
					= list of tuples where the first element of the tuple is the index of the first entity
		Return:
		"""		
		indicesToCut = findIndicesToCut(indices)
		for i in range(X.shape[0]):
			# example: (convolution formula(maxSenLen+2w-2, self.w)), self.numFilters)
			example = X[i]
			A = np.max(example[0:indicesToCut[i][0] + 1], axis = 0)
			B = np.max(example[indicesToCut[i][0] + 1, 
				indicesToCut[i][1] + 1], axis = 0)
			C = np.max(example[indicesToCut[i][1] + 1, :], axis = 0)
			# curResult: (3*self.numFilters,)
			curResult = np.concatenate(A, B, C)







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
	




