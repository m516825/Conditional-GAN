import tensorflow as tf
import numpy as np
import sys
import os
import data_utils
import _pickle as cPickle
from data_utils import VocabularyProcessor
from data_utils import Data
from WGAN import WGAN
from improved_WGAN import Improved_WGAN

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("hidden", 128, "hidden dimension of RNN hidden size")
tf.flags.DEFINE_integer("iter", 1000000, "number of training iter")
tf.flags.DEFINE_integer("z_dim", 100, "noise dimension")
tf.flags.DEFINE_integer("batch_size", 64, "batch size per iteration")
tf.flags.DEFINE_integer("display_every", 20, "predict model on dev set after this many steps (default: 200)")
tf.flags.DEFINE_integer("dump_every", 500, "predict model on dev set after this many steps (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 200)")

tf.flags.DEFINE_float("lr", 2e-4, "training learning rate")

tf.flags.DEFINE_string("img_dir", "./test_img/", "test image directory")
tf.flags.DEFINE_string("train_dir", "./MLDS_HW3_dataset/faces", "training data directory")
tf.flags.DEFINE_string("tag_path", "./MLDS_HW3_dataset/tags_clean.csv", "training data tags")
tf.flags.DEFINE_string("test_path", "./MLDS_HW3_dataset/sample_testing_text.txt", "sample test format")
tf.flags.DEFINE_string("checkpoint_file", "", "checkpoint_file to be load")
tf.flags.DEFINE_string("prepro_dir", "./prepro/", "tokenized train data's path")
tf.flags.DEFINE_string("vocab", "./vocab", "vocab processor path")
tf.flags.DEFINE_string("model", "Improved_WGAN", "init model name")

tf.flags.DEFINE_boolean("prepro", True, "preprocessing the training data")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def main(_):

	print("Parameters: ")
	for k, v in FLAGS.__flags.items():
		print("{} = {}".format(k, v))

	if not os.path.exists("./prepro/"):
		os.makedirs("./prepro/")

	if FLAGS.prepro:
		img_feat, tags_idx, a_tags_idx, vocab_processor = data_utils.load_train_data(FLAGS.train_dir, FLAGS.tag_path, FLAGS.prepro_dir, FLAGS.vocab)	
	else:
		img_feat = cPickle.load(open(os.path.join(FLAGS.prepro_dir, "img_feat.dat"), 'rb'))
		tags_idx = cPickle.load(open(os.path.join(FLAGS.prepro_dir, "tag_ids.dat"), 'rb'))
		a_tags_idx = cPickle.load(open(os.path.join(FLAGS.prepro_dir, "a_tag_ids.dat"), 'rb'))
		vocab_processor = VocabularyProcessor.restore(FLAGS.vocab)
	img_feat = np.array(img_feat, dtype='float32')/127.5 - 1.
	test_tags_idx = data_utils.load_test(FLAGS.test_path, vocab_processor)

	print("Image feature shape: {}".format(img_feat.shape))
	print("Tags index shape: {}".format(tags_idx.shape))
	print("Attribute Tags index shape: {}".format(a_tags_idx.shape))
	print("Vocab size: {}".format(len(vocab_processor._reverse_mapping)))
	print("Vocab max length: {}".format(vocab_processor.max_document_length))
	
	data = Data(img_feat, tags_idx, a_tags_idx, test_tags_idx, FLAGS.z_dim, vocab_processor)

	Model = getattr(sys.modules[__name__], FLAGS.model)	
	print(Model)

	model = Model(data, vocab_processor, FLAGS)
	
	model.build_model()
	
	model.train()

if __name__ == '__main__':
	tf.app.run()