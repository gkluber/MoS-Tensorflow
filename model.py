import math
import numpy as np
import tensorflow as tf

class MoS(object):
	
	
	def __init__(self, sess, ntoken, ninp, nhid, nhidlast, nlayers, dropout = 0.5, dropouth = 0.5,
					dropouti = 0.5, dropoute = 0.1, wdrop = 0, tie_weights = False,
					ldropout = 0.5, n_experts = 10):
		#Declare and define class variables
		
		#Tensorflow session
		self.sess = sess
		
		#Network architecture params
		self.ntokens = ntoken
		self.ninp = ninp
		self.nhid = nhid
		self.nhidlast = nhidlast
		self.nlayers = nlayers
		
		#Dropout params
		self.dropout = dropout
		self.dropouth = dropouth
		self.dropouti dropouti
		self.dropoute = dropoute
		self.wdrop = wdrop
		self.ldropout = ldropout
		
		#Misc
		self.tie_weights = tie_weights
		
		#The number of softmaxes to use in the mixture of softmaxes
		self.n_experts = n_experts
		
		#construct the Tensorflow graph for the model
		build_model()
		
	def build_model(self):
		#word embeddings--word2vec implementation
		self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
		#Noise Contrastive Estimation
		nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)
		embed_loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=nce_weights,
				biases=nce_biases,
				labels=train_labels,
				inputs=embed,
				num_sampled=num_sampled,
				num_classes=vocabulary_size))
		
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(embed_loss)
		
		cells = []
		for l in range(self.nlayers):
			cell = tf.contrib.rnn.GRUCell(num_units=self.nhid if l != self.nlayers - 1 else self.nhidlast)
			cells.append(cell)
		
		self.rnn_stack = tf.contrib.rnn.MultiRNNCell(cells)
		
		self.rnn_stack_output, self.stack_states = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)
		
		self.rnn_stack_output = tf.reshape(self.rnn_stack_output, [-1, self.nhid])
		self.rnn_stacked_outputs = tf.layers.dense(self.rnn_stack_output, self.nhidlast)
		
		#output size maps self.nhid to self.nhidlast
		self.rnn_stacked_outputs = tf.reshape(self.rnn_stacked_outputs, [-1, self.ninp, self.nhidlast])
		
		self.prior_logit = tf.contrib.layers.fully_connected(self.rnn_stacked_outputs, self.n_experts, activation_fn=None) #linear layers
		self.prior_logit = tf.reshape(self.prior_logit, [-1, self.n_experts])
		
		self.latent = tf.contrib.layers.fully_connected(self.rnn_stacked_outputs, self.n_experts, activation_fn=tf.nn.swish) #TODO: make activation customizable
		
		self.logits = tf.contrib.layers.fully_connected(self.latent, self.ninp, activation_fn=None)
		self.logits = tf.reshape(self.logit, [-1, self.ntokens])
		
		#computation of probabilities
		self.prob = tf.nn.softmax(logits)
		self.prob = tf.reshape(self.prob, [-1, self.n_experts, self.ntokens])
		