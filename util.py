import tensorflow as tf
import tflearn
import numpy as np

import nltk
from nltk.stem.lancaster import LancasterStemmer

from collections import Counter
import os

class Dictionary:
	def __init__(self, vocab_size):
		self.counter = Counter()
		self.word2id = {}
		self.id2word = []
		self.total = 0
		self.vocab_size = vocab_size
	
	#returns true until vocab_size reached
	def add_word(self, word) -> bool:
		if word not in self.word2id:
			if len(self.id2word) < vocab_size:
				self.id2word.append(word)
				self.word2id[word] = len(self.id2word) - 1
			else:
				return False
		token_id = word2id[word]
		self.counter[token_id] += 1
		self.total += 1
		return True
		
	def __len__(self):
		return len(self.word_dict)

def process_file(path, vocab_size):
	count = [['UNK', -1],['EOS',-1],[]]
	stemmer = LancasterStemmer()
	words = Dictionary(vocab_size)
	
	assert os.path.exists(path)
	
	with open(path,'r',encoding='utf-8') as f:
		for line in f:
			#tokenize each line
			line = tokenize(line)
			for token in set(line)
				words.add_word(token)
			
def tokenize(line):
	tokens = nltk.word_tokenize(line)
	#applies the stemmer (such that words have their suffixes dropped: tired -> tir, whine -> whin, whining -> whin)
	tokens = [stemmer.stem(w.lower()) for w in words]
	tokens.append("<eos>")
	return tokens