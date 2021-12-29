from __future__ import print_function
import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_io

from lpr.networks.lprnet import LPRNet
from tfutils.helpers import load_module, execute_mo

input_shape =(24, 94, 3)
num_classes = 37
rnn_cells_num= 128
max_lp_length = 20

def inference(rnn_cells_num, input, num_classes):
  cnn = LPRNet.lprnet(input)

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    classes = slim.conv2d(cnn, num_classes, [1, 13])
    pattern = slim.fully_connected(slim.flatten(classes), rnn_cells_num)  # patterns number
    width = int(cnn.get_shape()[2])
    pattern = tf.reshape(pattern, (-1, 1, 1, rnn_cells_num))
    pattern = tf.tile(pattern, [1, 1, width, 1])
    # pattern = slim.fully_connected(pattern, num_classes * width, normalizer_fn=None, activation_fn=tf.nn.sigmoid)
    # pattern = tf.reshape(pattern, (-1, 1, width, num_classes))

  inf = tf.concat(axis=3, values=[classes, pattern])  # skip connection over RNN
  inf = slim.conv2d(inf, num_classes, [1, 1], normalizer_fn=None,
                    activation_fn=None)  # fully convolutional linear activation

  inf = tf.squeeze(inf, [1])

  return inf
def parse_args():
  parser = argparse.ArgumentParser(description='Export model in IE format')
  parser.add_argument('--output_dir', default=None, help='Output Directory', required=True, type=str)
  parser.add_argument('--checkpoint', default=None, help='Default: latest', required=True, type=str)
  return parser.parse_args()



def freezing_graph( checkpoint, output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  shape = (None,) + tuple(input_shape) # NHWC, dynamic batch
  graph = tf.Graph()
  with graph.as_default():
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
      input_tensor = tf.placeholder(dtype=tf.float32, shape=shape, name='input')
      prob = inference(rnn_cells_num, input_tensor, num_classes)
      prob = tf.transpose(prob, (1, 0, 2))
      data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])
      result = tf.nn.ctc_greedy_decoder(prob, data_length, merge_repeated=True)
      predictions = tf.to_int32(result[0][0])
      tf.sparse_to_dense(predictions.indices, [tf.shape(input_tensor, out_type=tf.int64)[0], max_lp_length],
                         predictions.values, default_value=-1, name='d_predictions')
      init = tf.initialize_all_variables()
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  sess = tf.Session(graph=graph)
  sess.run(init)
  saver.restore(sess, checkpoint)
  frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["d_predictions"])
  tf.train.write_graph(sess.graph, output_dir, 'graph.pbtxt', as_text=True)
  path_to_frozen_model = graph_io.write_graph(frozen, output_dir, 'graph.pb.frozen', as_text=False)
  return path_to_frozen_model

def main(_):
  args = parse_args()

  checkpoint = args.checkpoint
  if not checkpoint or not os.path.isfile(checkpoint + '.index'):
    raise FileNotFoundError(str(checkpoint))

  step = checkpoint.split('.')[-2].split('-')[-1]
  output_dir = args.output_dir

  # Freezing graph
  frozen_dir = os.path.join(output_dir, 'frozen_graph')
  frozen_graph = freezing_graph(checkpoint, frozen_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
