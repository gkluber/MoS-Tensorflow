import os
import tensorflow as tf
import numpy as np

from collections import Counter


class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []
		self.counter = Counter()
		self.total = 0

    def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
		self.word2idx[word] = len(self.idx2word) - 1
		token_id = self.word2idx[word]
		self.counter[token_id] += 1
		self.total += 1
		return self.word2idx[word]

	def __len__(self):
		return len(self.idx2word)

class Corpus(object):
	def __init__(self, path):
		self.dictionary = Dictionary()
		self.train = self.tokenize(os.path.join(path, 'train.txt'))
		self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
		self.test = self.tokenize(os.path.join(path, 'test.txt'))

	def tokenize(self, path):
		"""Tokenizes a text file."""
		assert os.path.exists(path)
		# Add words to the dictionary
		with open(path, 'r', encoding='utf-8') as f:
			tokens = 0
			for line in f:
				words = line.split() + ['<eos>']
				tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)

		# Tokenize file content
		with open(path, 'r', encoding='utf-8') as f:
			id_vec = np.zeros((1,tokens), dtype=np.uint32)
			token = 0
			for line in f:
				words = line.split() + ['<eos>']
				for word in words:
					id_vec[:,token] = self.dictionary.word2idx[word]
					token += 1
		
		return id_vec

'''#lazy file streamer to read the corpus in chunks without loading it all into memory at once
def stream(file, chunk_lines = 10000):
	while True:
		data = reduce(lambda x,y: x+y, [file.readLine()]*chunk_lines)
		if not data:
			break
		yield data 

'''