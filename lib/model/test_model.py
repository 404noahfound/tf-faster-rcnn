from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model.config import cfg

def test_model_graph(net):
  # TODO: maintain num_classes
  num_classes = 100
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True
  with tf.Session(config=tfconfig) as sess:
    with sess.graph.as_default():
      tf.set_random_seed(cfg.RNG_SEED)
      layers = net.create_architecture(sess, 'TRAIN', num_classes, tag='default',
                                               anchor_scales=cfg.ANCHOR_SCALES,
                                               anchor_ratios=cfg.ANCHOR_RATIOS)
      loss = layers['total_loss']
  return loss
