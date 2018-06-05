import tensorflow as tf
from tf_utils import loadData

def main(unused_argv):
	print("Hello world")

	# Load the data
	embedding_matrix, word2Index, index2Word = loadData()

	# Initialize model
	#model = RelationClassifier()
	with tf.Session() as sess:

		# Initialize variables
		sess.run(tf.global_variables_initializer())
		#print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())

		# Train
		



if __name__ == "__main__":
    tf.app.run()
    print("I am exiting if name main")