import numpy as np
import sys
import os
import csv
from scipy import misc
import collections
import _pickle as cPickle
from tensorflow.python.platform import gfile
import scipy.stats as stats
import math
import random
import copy

try:
	import cPickle as pickle
except ImportError:
	import pickle

length_limit = 25
vocab_limit = 200
topk = 5

class Data(object):
	def __init__(self, img_feat, tags_idx, a_tags_idx, test_tags_idx, z_dim, vocab_processor):
		self.z_sampler = stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)
		self.length = len(tags_idx)
		self.current = 0
		self.img_feat = img_feat
		self.tags_idx = tags_idx
		self.a_tags_idx = a_tags_idx
		self.w_idx = np.arange(self.length)
		self.w_idx2 = np.arange(self.length)
		self.tmp = 0
		self.epoch = 0
		self.vocab_processor = vocab_processor
		self.vocab_size = len(vocab_processor._reverse_mapping)
		self.unk_id = vocab_processor._mapping['<UNK>']
		self.eos_id = vocab_processor._mapping['<EOS>']
		self.hair_id = vocab_processor._mapping['hair']
		self.eyes_id = vocab_processor._mapping['eyes']
		self.gen_info()
		self.test_tags_idx = self.gen_test_hot(test_tags_idx)
		self.fixed_z = self.next_noise_batch(len(self.test_tags_idx), z_dim)

		idx = np.random.permutation(np.arange(self.length))
		self.w_idx2 = self.w_idx2[idx]

	def gen_test_hot(self, test_intput):
		test_hot = []
		for tag in test_intput:
			eyes_hot = np.zeros([len(self.eyes_idx)])
			eyes_hot[np.where(self.eyes_idx == tag[2])[0]] = 1
			hair_hot = np.zeros([len(self.hair_idx)])
			hair_hot[np.where(self.hair_idx == tag[0])[0]] = 1
			tag_vec = np.concatenate((eyes_hot, hair_hot))
			test_hot.append(tag_vec)

		return np.array(test_hot)

	def gen_info(self):
		self.eyes_idx = np.array([idx for idx in set(self.a_tags_idx[:,0])])
		self.hair_idx = np.array([idx for idx in set(self.a_tags_idx[:,1])])
		self.type = []
		for a_tag in self.a_tags_idx:
			if a_tag[0] == self.unk_id:
				self.type.append(1)
			elif a_tag[1] == self.unk_id:
				self.type.append(2)
			else:
				self.type.append(0)
		self.type = np.array(self.type)

		self.one_hot = []
		for a_tag in self.a_tags_idx:
			eyes_hot = np.zeros([len(self.eyes_idx)])
			eyes_hot[np.where(self.eyes_idx == a_tag[0])[0]] = 1
			hair_hot = np.zeros([len(self.hair_idx)])
			hair_hot[np.where(self.hair_idx == a_tag[1])[0]] = 1
			tag_vec = np.concatenate((eyes_hot, hair_hot))
			self.one_hot.append(tag_vec)
		self.one_hot = np.array(self.one_hot)

	def next_data_batch(self, size, neg_sample=False):
		if self.current == 0:
			self.epoch += 1
			idx = np.random.permutation(np.arange(self.length))
			self.img_feat = self.img_feat[idx]
			self.tags_idx = self.tags_idx[idx]
			self.a_tags_idx = self.a_tags_idx[idx]
			self.type = self.type[idx]
			self.one_hot = self.one_hot[idx]
			idx = np.random.permutation(np.arange(self.length))
			self.w_idx = self.w_idx[idx]

		if self.current + size < self.length:
			img, tags, a_tags, d_t, widx, hot = self.img_feat[self.current:self.current+size], self.tags_idx[self.current:self.current+size], self.a_tags_idx[self.current:self.current+size], self.type[self.current:self.current+size], self.w_idx[self.current:self.current+size], self.one_hot[self.current:self.current+size]
			self.current += size

		else:
			img, tags, a_tags, d_t, widx, hot = self.img_feat[self.current:], self.tags_idx[self.current:], self.a_tags_idx[self.current:], self.type[self.current:], self.w_idx[self.current:], self.one_hot[self.current:]
			self.current = 0

		size = len(tags)
		type0_idx = np.where(d_t == 0)[0]
		if len(type0_idx) > 0:
			while True:
				mis_idx = np.where(np.mean(np.equal(a_tags[type0_idx], self.a_tags_idx[widx][type0_idx]), axis=1) == 1)[0]
				if len(mis_idx) == 0:
					break
				if self.tmp + len(mis_idx) >= self.length:
					idx = np.random.permutation(np.arange(self.length))
					self.w_idx2 = self.w_idx2[idx]
					self.tmp = 0
				widx[type0_idx[mis_idx]] = self.w_idx2[self.tmp:self.tmp+len(mis_idx)]
				self.tmp += len(mis_idx)

		# eye:unk, hair:tag
		type1_idx = np.where(d_t == 1)[0]
		if len(type1_idx) > 0:
			while True:
				mis_idx = np.where(np.equal(a_tags[type1_idx][:,1], self.a_tags_idx[widx][type1_idx,1]) == True)[0]
				if len(mis_idx) == 0:
					break
				if self.tmp + len(mis_idx) >= self.length:
					idx = np.random.permutation(np.arange(self.length))
					self.w_idx2 = self.w_idx2[idx]
					self.tmp = 0
				widx[type1_idx[mis_idx]] = self.w_idx2[self.tmp:self.tmp+len(mis_idx)]
				self.tmp += len(mis_idx)

		# eye:tag, hair:unk
		type2_idx = np.where(d_t == 2)[0]
		if len(type2_idx) > 0:
			while True:
				mis_idx = np.where(np.equal(a_tags[type2_idx][:,0], self.a_tags_idx[widx][type2_idx,0]) == True)[0]
				if len(mis_idx) == 0:
					break
				if self.tmp + len(mis_idx) >= self.length:
					idx = np.random.permutation(np.arange(self.length))
					self.w_idx2 = self.w_idx2[idx]
					self.tmp = 0
				widx[type2_idx[mis_idx]] = self.w_idx2[self.tmp:self.tmp+len(mis_idx)]
				self.tmp += len(mis_idx)

		return img, hot, a_tags, self.img_feat[widx], self.one_hot[widx]

	def next_noise_batch(self, size, dim):
		return self.z_sampler.rvs([size, dim]) #np.random.uniform(-1.0, 1.0, [size, dim])

class VocabularyProcessor(object):
	def __init__(self, max_document_length, vocabulary, unknown_limit=float('Inf'), drop=False):
		self.max_document_length = max_document_length
		self._reverse_mapping = ['<UNK>', '<EOS>'] + vocabulary
		self.make_mapping()
		self.unknown_limit = unknown_limit
		self.drop = drop

	def make_mapping(self):
		self._mapping = {}
		for i, vocab in enumerate(self._reverse_mapping):
			self._mapping[vocab] = i

	def transform(self, raw_documents, length=-1):
		data = []
		lengths = []
		seq_length = self.max_document_length if length < 0 else length
		for tokens in raw_documents:
			word_ids = np.ones(seq_length, np.int32) * self._mapping['<EOS>']
			length = 0
			unknown = 0
			if self.drop and len(tokens.split()) > seq_length:
				continue
			for idx, token in enumerate(tokens.split()):
				if idx >= seq_length:
					break
				word_ids[idx] = self._mapping.get(token, 0)
				length = idx
				if word_ids[idx] == 0:
					unknown += 1
			length = length+1
			if unknown <= self.unknown_limit:
				data.append(word_ids)
				lengths.append(length)

		data = np.array(data)
		lengths = np.array(lengths)

		return np.array(data)
			
	def save(self, filename):
		with gfile.Open(filename, 'wb') as f:
			f.write(pickle.dumps(self))
	@classmethod
	def restore(cls, filename):
		with gfile.Open(filename, 'rb') as f:
			return pickle.loads(f.read())

def load_train_data(img_dir, tag_path, prepro_dir, vocab_path, shuffle_time=1):

	vocab = collections.defaultdict(int)
	used_vocab = collections.defaultdict(int)
	raw_tags = []
	attrib_tags = [] 
	img_feat = []
	with open(tag_path, 'r') as f:
		for ridx, row in enumerate(csv.reader(f)):
			tags = row[1].split('\t')
			for t in tags:
				tag = t.split(':')[0].strip()
				for w in tag.split():
					vocab[w] += 1

	with open(tag_path, 'r') as f:
		for ridx, row in enumerate(csv.reader(f)):
			tags = row[1].split('\t')
			c_tags = []
			k_tags = {}
			length = 0
			has_attrib = False
			attrib = {'eyes':'<UNK>', 'hair':'<UNK>'}
			for t in tags:
				if t != '':
					tag = t.split(':')[0].strip()
					s_tag = tag.split()
					if len(s_tag) > 2:
						continue
					score = int(t.split(':')[1].strip())
					C_flag = False
					B_flag = False
					for w in s_tag:
						if w == 'hair' and (s_tag[0] == 'long' or s_tag[0] == 'short'):
							C_flag = True
							break
						if w == 'eyes' or w == 'hair':
							if attrib[w] != '<UNK>':
								B_flag = True
								break
							attrib[w] = s_tag[0]
							has_attrib = True
							score = float('Inf')
						if vocab[w] < vocab_limit:
							C_flag = True
							break
					if C_flag:
						continue
					if B_flag:
						break
					length += len(s_tag)
					k_tags[tag] = score

			if len(k_tags) == 0 or not has_attrib or B_flag:
				continue

			a_tags = ' '.join([attrib['eyes'], attrib['hair']])
					
			for idx, (k, v) in enumerate(sorted(k_tags.items(), key=lambda x:x[1], reverse=True)):
				if idx < topk:
					c_tags.append(k)
					for w in k.split():
						used_vocab[w] += 1

			c_tags = [attrib['eyes'] + ' eyes', attrib['hair'] + ' hair']

			img_path = os.path.join(img_dir, '{}.jpg'.format(ridx))
			feat = misc.imread(img_path)
			feat = misc.imresize(feat, [64, 64, 3])
			random.shuffle(c_tags)
			raw_tags.append(' '.join(c_tags))
			attrib_tags.append(a_tags)
			img_feat.append(feat)

			m_feat = np.fliplr(feat)
			random.shuffle(c_tags)
			raw_tags.append(' '.join(c_tags))
			attrib_tags.append(a_tags)
			img_feat.append(m_feat)

			feat_p5 = misc.imrotate(feat, 5)
			random.shuffle(c_tags)
			raw_tags.append(' '.join(c_tags))
			attrib_tags.append(a_tags)
			img_feat.append(feat_p5)

			feat_m5 = misc.imrotate(feat, -5)
			random.shuffle(c_tags)
			raw_tags.append(' '.join(c_tags))
			attrib_tags.append(a_tags)
			img_feat.append(feat_m5)

	img_feat = np.array(img_feat)

	vocabulary = []
	for k, v in sorted(used_vocab.items(), key=lambda x:x[1], reverse=True):
		vocabulary.append(k)

	avg_length = sum([len(tags.split()) for tags in raw_tags])/len(raw_tags)
	max_length = max([len(tags.split()) for tags in raw_tags])
	vocab_processor = VocabularyProcessor(max_document_length=max_length, vocabulary=vocabulary)
	tags_idx = vocab_processor.transform(raw_tags)
	a_tags_idx = vocab_processor.transform(attrib_tags, 2)

	print("max sentence length: {}".format(max_length))
	print("avg sentence length: {}".format(avg_length))

	cPickle.dump(img_feat, open(os.path.join(prepro_dir, "img_feat.dat"), 'wb'))
	cPickle.dump(tags_idx, open(os.path.join(prepro_dir, "tag_ids.dat"), 'wb'))
	cPickle.dump(a_tags_idx, open(os.path.join(prepro_dir, "a_tag_ids.dat"), 'wb'))

	vocab_processor.save(vocab_path)

	return img_feat, tags_idx, a_tags_idx, vocab_processor

def load_test(test_path, vocab_processor):
	test = []
	with open(test_path, 'r') as f:
		for line in f.readlines():
			line = line.strip().split(',')[1]
			test.append(line)
	tags_idx = vocab_processor.transform(test)

	return  tags_idx

def dump_img(img_dir, img_feats, iters):
	if not os.path.exists(img_dir):
		os.makedirs(img_dir)
	
	img_feats = (img_feats + 1.)/2 * 255.
	img_feats = np.array(img_feats, dtype=np.uint8)

	for idx, img_feat in enumerate(img_feats):
		path = os.path.join(img_dir, 'iters_{}_test_{}.jpg'.format(iters, idx))
		misc.imsave(path, img_feat)






