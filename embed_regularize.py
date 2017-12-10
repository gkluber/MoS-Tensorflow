import numpy as np
import tensorflow as tf


def embedded_dropout(embedding: tf.Variable, words: tf.Variable, dropout=0.1, scale=None) -> tf.Operation:
	