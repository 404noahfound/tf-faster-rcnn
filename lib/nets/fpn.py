# TODO:
# - a net that receives head and feature_maps
# - how to accumulate the result of different stages? add them all and pick or
# pick then add them all?
# - merge the input before head, use only one head
# - add fpn to rpn first, then rcnn
# - replace the multi-scale anchors with single scale
# - how to handle inputs to different stages?

import tensorflow as tf
import tensorflow.contrib.slim as slim
from model.config import cfg

class FeaturePyramidNetwork():

  def __init__(self, base_net, is_training, name='pyramid'):
    self._name = name
    self._base_net = base_net
    self._layers = {}
    self._input_layers = {}
    self._heads = {}
    self._stage_outputs = {}
    self._merge_outputs = {}
    self._output_name_list = []
    self._is_training = is_training

    if base_net._resnet_scope is not None:
      self._base_net_scope = base_net._resnet_scope
    else:
      raise NotImplementedError
    assert self._base_net_scope == 'resnet_v1_50'

    # assume that network maps are always named as 'C%d'
    self._net_map = {}
    self._net_begin = None

  def build_net(self):
    self._load_input_layers()
    self.build_pyramid()
    self.build_heads()
    self.merge_outputs()
    return self._merge_outputs

  def _load_input_layers(self):
    input_layers = {}
    for key in self._net_map:
      input_layers[key] = self._base_net.get_layer(self._net_map[key])
    self._input_layers = input_layers


  def build_pyramid(self):

    initializer, _ = self.set_initializers()
    self._load_input_layers()
    is_training = self._is_training
    net_begin = self._net_begin
    net_end = len(self._net_map) + net_begin - 1

    with tf.variable_scope(self._name):
      self._layers['P%d'%net_end] = \
        slim.conv2d(self._input_layers['C%d'%net_end], 256, [1, 1], stride=1,
          scope='C%d'%net_end, trainable=is_training,
          weights_initializer=initializer)
      for c in range (net_end - 1, net_begin - 1, -1):
        s, s_ = self._layers['P%d'%(c+1)], self._input_layers['C%d'%(c)]
        up_shape = tf.shape(s_)
        s = tf.image.resize_bilinear(s, [up_shape[1], up_shape[2]],
          name='P%d/upscale'%(c+1))
        s_ = slim.conv2d(s_, 256, [1,1], stride=1, scope='C%d'%c,
          trainable=is_training, weights_initializer=initializer)
        s = tf.add(s, s_, name='C%d/addition'%c)
        s = slim.conv2d(s, 256, [3,3], stride=1, scope='C%d/fusion'%c,
          trainable=is_training, weights_initializer=initializer)
        self._layers['P%d'%c] = s

    assert len(self._layers) == len(self._input_layers)
    return self._layers

  def build_heads(self, head_builder, head_name):
    raise NotImplementedError('child class must implement this method')
    # this is only a template
    scope = self._name + '/' + head_name
    with tf.variable_scope(scope):
      for layer_key in self._layers:
        layer = self._layers[layer_key]
        with tf.variable_scope(layer_key):
          head, output = head_builder(layer)
          self._heads[layer_key] = head
          self._stage_outputs[layer_key] = output

  def set_initializers(self):
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    return initializer, initializer_bbox

  def merge_outputs(self):
    for output_name in self._output_name_list:
      print('merging output for %s' % output_name)
      outputs = [self._stage_outputs[stage_name][output_name] \
        for stage_name in self._stage_outputs]
        print('unmerged output size for stage {}: {}'.format\
        (stage_name, self._stage_outputs[stage_name][output_name].get_shape()))
      outputs = tf.concat(values=outputs, axis=0)
      self._merge_outputs[output_name] = outputs
      print('merged output size: {}'.format(outputs.get_shape()))
