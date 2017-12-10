import os, sys
import time
import math
import numpy as np
import tensorflow as tf

#garbage collector
import gc

import data
import model

flags = tf.app.flags
flags.DEFINE_string("data", "./penn/", "Path to the folder containing the corpus. Should contain a test.txt, valid.txt, and train.txt file")
flags.DEFINE_string("model", "LSTM", "The type of recurrent cell to use. Options: [RNN_TANH, RNN_RELU, LSTM, GRU, SRU, QRNN_TANH, QRNN_RELU, RNN_SWISH, QRNN_SWISH]")
flags.DEFINE_string("save_path", "The path to save the final model at")
flags.DEFINE_integer("emsize", 400, "The size of the word embeddings")
flags.DEFINE_integer("n_hidden", 1150, "The number of hidden units per layer")
flags.DEFINE_integer("n_hidden_last", -1, "The number of hidden units for the last RNN layer")
flags.DEFINE_integer("n_layers", 3, "The number of layers in the RNN stack")
flags.DEFINE_integer("epochs", 8000, "The maximum number of epochs to train on")
flags.DEFINE_integer("batch_size", 20, "The maximum number of epochs to train on")
flags.DEFINE_integer("bptt", 70, "The sequence length")
flags.DEFINE_integer("seed", 1111, "Random seed")
flags.DEFINE_integer("nonmono", 5, "Random seed")
flags.DEFINE_integer("log_interval", 200, "Report interval")
flags.DEFINE_integer("n_experts", 10, "Number of experts--the number of softmaxes that the MoS is composed from")
flags.DEFINE_integer("max_seq_len_delta", 40, "Maximum sequence length")
flags.DEFINE_float("lr", 30, "Initial learning rate")
flags.DEFINE_float("clip", 0.25, "Gradient clipping threshold")
flags.DEFINE_float("dropout", 0.4, "Dropout applied to the layers (0 = no dropout)")
flags.DEFINE_float("dropouth", 0.3, "Droput applied to the RNN layers (0 = no dropout)")
flags.DEFINE_float("dropouti", 0.65, "Dropout applied to the embedding layers (0 = no dropout)")
flags.DEFINE_float("dropoute", 0.1, "Dropout to remove words from the embedding layer (0 = no dropout)")
flags.DEFINE_float("dropoutl", 0.2, "Dropout applied to layers (0 = no dropout)")
flags.DEFINE_float("alpha", 2, "Alpha L2 regularization on the RNN objective (alpha = 0 means no regularization)")
flags.DEFINE_float("beta", 1, "Beta slowness regularization applied to the RNN activation (beta = 0 means no regularization)")
flags.DEFINE_float("wdecay", 1.2e-6, "Weight decay applied to all weights in the network")
flags.DEFINE_boolean("tied_weights", False, "Ties the word embedding and softmax weights")
flags.DEFINE_boolean("continue_train", True, "Continue training from the last checkpoint, if any")
FLAGS = flags.FLAGS

def main(_):

	print(FLAGS.__flags)
	
	if not os.path.exists(FLAGS.save_path):
		os.makedirs(FLAGS.save_path)
	
	if FLAGS.nhidlast < 0:
		FLAGS.nhidlast = FLAGS.emsize
	if FLAGS.dropoutl < 0:
		FLAGS.dropoutl = FLAGS.dropouth
	
	with tf.Session() as sess:
		model = MoS(sess, FLAGS.test_render, FLAGS.ignore_checkpoint, FLAGS.manual, FLAGS.save,
							FLAGS.learning_rate, FLAGS.beta1, FLAGS.beta2, FLAGS.discount_rate, FLAGS.epochs,
							FLAGS.max_steps, FLAGS.games_per_update, FLAGS.save_iterations, FLAGS.test_games,
							FLAGS.checkpoint_dir)
		
		if FLAGS.train:
			model.train()
		else:
			model.load_model()
			model.test()

if __name__ == '__main__':
	tf.app.run()