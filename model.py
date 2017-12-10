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
		self.ntokens = ntokens
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
		self.encoder = tf.Variable(tf.random_uniform([self.ntokens, self.ninp], -1.0, 1.0)
		
		cells = []
		for l in range(self.nlayers):
			cell = tf.contrib.rnn.GRUCell(num_units=self.nhid if l != self.nlayers - 1 else self.nhidlast)
			cells.append(cell)
		
		self.rnn_stack = tf.contrib.rnn.MultiRNNCell(cells)
		
		self.rnn_stack_output, self.stack_states = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)
		
		self.rnn_stacked_rnn_outputs = tf.reshape(self.rnn_stack_output, [-1, self.nhid])
		self.rnn_stacked_outputs = tf.layers.dense(self.rnn_stacked_rnn_outputs, self.nhidlast)
		
		#output size maps self.nhid to self.nhidlast
		self.stack_output = tf.reshape(self.rnn_stacked_outputs, [-1, self.ninp, self.nhidlast])
		
		
		
	