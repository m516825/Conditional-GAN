import tensorflow as tf
import tensorflow.contrib as tc
import math

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class Generator(object):
	def __init__(self, 
		max_seq_length, 
		vocab_size, 
		embedding_size, 
		hidden_size, 
		img_row, 
		img_col):

		self.max_seq_length = max_seq_length
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.img_row = img_row
		self.img_col = img_col
		
	def __call__(self, seq_idx, z, reuse=False, train=True):

		batch_size = tf.shape(seq_idx)[0]

		tags_vectors = seq_idx

		with tf.variable_scope("g_net") as scope:

			if reuse:
				scope.reuse_variables()

			noise_vector = tf.concat([tags_vectors, z], axis=1)

			fc2 = tc.layers.fully_connected(
				noise_vector, 4*4*256,
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			fc2 = tf.layers.batch_normalization(fc2, training=train)
			fc2 = tf.reshape(fc2, [-1, 4, 4, 256])
			fc2 = tf.nn.relu(fc2)

			conv1 = tc.layers.convolution2d_transpose(
				fc2, 128, [5, 5], [2, 2],
				padding='same',
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			conv1 = tf.layers.batch_normalization(conv1, training=train)
			conv1 = tf.nn.relu(conv1)

			conv2 = tc.layers.convolution2d_transpose(
				conv1, 64, [5, 5], [2, 2],
				padding='same',
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			conv2 = tf.layers.batch_normalization(conv2, training=train)
			conv2 = tf.nn.relu(conv2)

			conv3 = tc.layers.convolution2d_transpose(
				conv2, 32, [5, 5], [2, 2],
				padding='same',
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			conv3 = tf.layers.batch_normalization(conv3, training=train)
			conv3 = tf.nn.relu(conv3)

			conv4 = tc.layers.convolution2d_transpose(
				conv3, 3, [5, 5], [2, 2],
				padding='same',
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			conv4 = tf.nn.tanh(conv4)

			return conv4

	@property
	def vars(self):
		return [var for var in tf.global_variables() if "g_net" in var.name]

class Discriminator(object):
	def __init__(self, 
		max_seq_length, 
		vocab_size, 
		embedding_size, 
		hidden_size,
		img_row,
		img_col):

		self.max_seq_length = max_seq_length
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.img_row = img_row
		self.img_col = img_col
		
	def __call__(self, seq_idx, img, reuse=True):

		batch_size = tf.shape(seq_idx)[0]

		tags_vectors = seq_idx

		with tf.variable_scope("d_net") as scope:

			if reuse == True:
				scope.reuse_variables()

			conv1 = tc.layers.convolution2d(
				img, 32, [5, 5], [2, 2],
				padding='same',
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			conv1 = tf.layers.batch_normalization(conv1, training=True)
			conv1 = leaky_relu(conv1)

			conv2 = tc.layers.convolution2d(
				conv1, 64, [5, 5], [2, 2],
				padding='same',
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			conv2 = tf.layers.batch_normalization(conv2, training=True)
			conv2 = leaky_relu(conv2)
			
			conv3 = tc.layers.convolution2d(
				conv2, 128, [5, 5], [2, 2],
				padding='same',
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			conv3 = tf.layers.batch_normalization(conv3, training=True)
			conv3 = leaky_relu(conv3)

			tags_vectors = tf.expand_dims(tf.expand_dims(tags_vectors, 1), 2)
			tags_vectors = tf.tile(tags_vectors, [1, 8, 8, 1])

			condition_info = tf.concat([conv3, tags_vectors], axis=-1)

			conv4 = tc.layers.convolution2d(
				condition_info, 128, [1, 1], [1, 1],
				padding='same',
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			conv4 = tf.layers.batch_normalization(conv4, training=True)
			conv4 = leaky_relu(conv4)

			conv5 = tc.layers.convolution2d(
				conv4, 1, [8, 8], [1, 1],
				padding='valid',
				weights_initializer=tf.random_normal_initializer(stddev=0.02),
				activation_fn=None
				)
			output = tf.squeeze(conv5, [1, 2, 3])

			return output
	@property
	def vars(self):
		return [var for var in tf.global_variables() if "d_net" in var.name]


