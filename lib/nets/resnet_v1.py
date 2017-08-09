# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np

from nets.network import Network
from model.config import cfg
from nets.rpn_fpn import RPN_FPN

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    'is_training': cfg.TRAIN.BN_TRAIN and is_training,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': cfg.TRAIN.BN_TRAIN,
    'updates_collections': tf.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

class resnetv1(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._num_layers = num_layers
    self._resnet_scope = 'resnet_v1_%d' % num_layers

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.RESNET.MAX_POOL:
        pre_pool_size = cfg.POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE],
                                         name="crops")
    return crops

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def build_base(self):
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

    return net

  def build_resnet_blocks(self):
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                # use stride 1 for the last conv4 layer
                resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    elif self._num_layers == 101:
      blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                # use stride 1 for the last conv4 layer
                resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
                resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    elif self._num_layers == 152:
      blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
                # use stride 1 for the last conv4 layer
                resnet_v1_block('block3', base_depth=256, num_units=36, stride=1),
                resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    else:
      # other numbers are not supported
      raise NotImplementedError
    self._resnet_blocks = blocks
    return blocks

  def build_resnet(self):
    # TODO:
    # issue: the two parts have different scope names!
    # set reuse = True does not work.
    is_training = self._is_training
    blocks = self.build_resnet_blocks()
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
      net_conv = self.build_base()

    if cfg.RESNET.FIXED_BLOCKS > 0:
      raise NotImplementedError('support for FIXED_BLOCKS is removed')
      # with slim.arg_scope(resnet_arg_scope(is_training=False)):
      #   net_conv, end_points = resnet_v1.resnet_v1(net_conv,
      #                                blocks[0:cfg.RESNET.FIXED_BLOCKS],
      #                                global_pool=False,
      #                                include_root_block=False,
      #                                reuse=True,
      #                                scope=self._resnet_scope)
    if cfg.RESNET.FIXED_BLOCKS < 3:
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv, end_points = resnet_v1.resnet_v1(net_conv,
                                           blocks,
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    self._act_summaries.append(net_conv)
    return net_conv, end_points

  def set_initializers(self):
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    return initializer, initializer_bbox

  def build_rpn(self, initializer):
    if cfg.USE_RPN_FPN:
      fpn = RPN_FPN(self)
      outputs = fpn.build_net()
      self._rpn_fpn = fpn
      return self._predictions['rois']
    is_training = self._is_training
    net_conv = self._layers['head']
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # build the anchors for the image
      self._anchor_component()
      rpn = slim.conv2d(net_conv, 512, [3, 3], trainable=is_training,
                        weights_initializer=initializer,
                        scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1],
                                  trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None,
                                  scope='rpn_cls_score')
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2,
                                                  'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape,
                                                 "rpn_cls_prob_reshape")
      rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]),
                                          axis=1, name="rpn_cls_pred")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape,
                                         self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1],
                                  trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None,
                                  scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a deterministic order for the computing graph,
        # for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois

  def build_rcnn(self, initializer, initializer_bbox):
    is_training = self._is_training
    net_conv = self._layers['head']
    blocks = self._resnet_blocks
    rois = self._predictions["rois"]

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
      else:
        raise NotImplementedError
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      fc7, _ = resnet_v1.resnet_v1(pool5,
                                   blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   scope=self._resnet_scope + '_res_conv')

      # this line is only for test, delete it!
      print('shape of pool5: {}'.format(pool5.get_shape()))
    #   fc7 = slim.fully_connected(pool5, 1024)
      print('shape of fc7: {}'.format(fc7.get_shape()))

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # Average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
      print('shape of fc7: {}'.format(fc7.get_shape()))
      cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer,
                                       trainable=is_training, activation_fn=None, scope='cls_score')
      cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred
    return cls_prob, bbox_pred

  def build_faster_rcnn_component(self):
    is_training = self._is_training
    initializer, initializer_bbox = self.set_initializers()
    rois = self.build_rpn(initializer)
    cls_prob, bbox_pred = self.build_rcnn(initializer, initializer_bbox)
    self._score_summaries.update(self._predictions)
    return rois, cls_prob, bbox_pred

  def get_layer(self, layer_name):
    assert layer_name in self._layers['end_points'], \
      '%s is not in the end_points'%layer_name
    return self._layers['end_points'][layer_name]

  def build_network(self, sess, is_training=True):

    self._is_training = is_training
    assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 3)
    # Now the base is always fixed during training

    self._layers['head'], self._layers['end_points'] = self.build_resnet()

    rois,cls_prob,bbox_pred = self.build_faster_rcnn_component()
    # for pred in self._predictions:
    #   print('shape of prediction {} is {}'.format(pred, self._predictions[pred].get_shape()))

    return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._resnet_scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                           tf.reverse(conv1_rgb, [2])))
