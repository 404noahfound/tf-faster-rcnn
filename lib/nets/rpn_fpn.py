import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.fpn import FeaturePyramidNetwork

class RPN_FPN(FeaturePyramidNetwork):

  def __init__(self, base_net, name='rpn_fpn'):
    FeaturePyramidNetwork.__init__(self, base_net, name)

    self._output_name_list= {}
    self._output_name_list['predictions'] = \
      {"rpn_cls_score": 1, 'rpn_cls_score_reshape': 1, 'rpn_cls_prob': 1,
        'rpn_cls_pred': 0, 'rpn_bbox_pred': -1, 'rois': 0}
    self._output_name_list['proposal_targets'] = \
      {'rois': 0, 'labels': 0, 'bbox_targets': 0, 'bbox_inside_weights': 0,
       'bbox_outside_weights': 0}
    self._output_name_list['anchor_targets'] = \
      {'rpn_labels': 0, 'rpn_bbox_targets': 0,
       'rpn_bbox_inside_weights': 0, 'rpn_bbox_outside_weights': 0}
    for name in self._output_name_list:
      self._stage_outputs[name] = \
        dict((n, {}) for n in self._output_name_list[name])
      self._merge_outputs[name] = \
        dict((n, None) for n in self._output_name_list[name])

    self._net_map = {
                  #  'C1':'resnet_v1_50/conv1/Relu:0',
                   'C2':'resnet_v1_50/block1/unit_2/bottleneck_v1',
                   'C3':'resnet_v1_50/block2/unit_3/bottleneck_v1',
                   'C4':'resnet_v1_50/block3/unit_5/bottleneck_v1',
                   'C5':'resnet_v1_50/block4/unit_3/bottleneck_v1',
      }
    self._stage_list = ['P2', 'P3', 'P4', 'P5']
    self._net_begin = 2

  def build_rpn_head(self, base_layer):
    base_net = self._base_net
    initializer, _ = self.set_initializers()
    is_training = self._is_training
    net_conv = base_layer
    # TODO:implement anchor_component
    num_anchors = base_net._num_anchors
    base_net._anchor_component()
    rpn = slim.conv2d(net_conv, 256, [3,3], trainable=is_training,
      weights_initializer=initializer, scope="rpn_conv/3x3")
    base_net._act_summaries.append(rpn)
    rpn_cls_score_raw = slim.conv2d(rpn, base_net._num_anchors * 2, [1, 1],
      trainable=is_training, weights_initializer=initializer, padding='VALID',
      activation_fn=None, scope='rpn_cls_score_raw')
    # TODO: implemnt reshape layer
    rpn_cls_score_reshape = base_net._reshape_layer(rpn_cls_score_raw, 2,
      'rpn_cls_score_reshape')
    rpn_cls_prob_raw = base_net._softmax_layer(rpn_cls_score_reshape,
      "rpn_cls_prob_raw")
    rpn_cls_prob_reshape = base_net._reshape_layer(rpn_cls_prob_raw,
      num_anchors * 2, 'rpn_cls_prob')
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]),
                                           axis=1, name="rpn_cls_pred")
    rpn_bbox_pred = slim.conv2d(rpn, num_anchors * 4, [1, 1],
                                 trainable=is_training,
                                 weights_initializer=initializer,
                                 padding='VALID', activation_fn=None,
                                 scope='rpn_bbox_pred')
    if is_training:
      rois, roi_scores = base_net._proposal_layer(rpn_cls_prob_reshape,
        rpn_bbox_pred, "rois")

      # TODO: figure out what this part is doing
      rpn_labels = base_net._anchor_target_layer(rpn_cls_score_raw, "anchor")
      # Try to have a deterministic order for the computing graph,
      # for reproducibility
      with tf.control_dependencies([rpn_labels]):
        rois, _ = base_net._proposal_target_layer(rois, roi_scores, "rpn_rois")

    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = base_net._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError
    predictions = {}
    predictions["rpn_cls_score"] = rpn_cls_score_raw
    predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    predictions["rpn_cls_prob"] = rpn_cls_prob_reshape
    predictions["rpn_cls_pred"] = rpn_cls_pred
    predictions["rpn_bbox_pred"] = rpn_bbox_pred
    predictions["rois"] = rois

    proposal_targets = base_net._proposal_targets
    anchor_targets = base_net._anchor_targets
    predictions = predictions
    outputs = {'predictions': predictions, 'proposal_targets': proposal_targets,
      'anchor_targets': anchor_targets}

    return rois, outputs


  def build_heads(self):
    scope = self._name + '/RPN_FPN'
    with tf.variable_scope(scope):
      for layer_key in self._layers:
        layer = self._layers[layer_key]
        with tf.variable_scope(layer_key):
          head, outputs = self.build_rpn_head(layer)
          self._heads[layer_key] = head
          for output_group in outputs:
            for output_name in outputs[output_group]:
              self._stage_outputs[output_group][output_name][layer_key] = \
                outputs[output_group][output_name]

  def merge_outputs(self):
    base_net = self._base_net

    for output_group in self._output_name_list:
      with tf.variable_scope('outputs/'+output_group):
        self.merger(
          self._output_name_list[output_group],
          self._stage_outputs[output_group],
          self._merge_outputs[output_group]
        )
    base_net._anchor_targets = self._merge_outputs['anchor_targets']
    base_net._proposal_targets = self._merge_outputs['proposal_targets']
    base_net._predictions = self._merge_outputs['predictions']
    base_net._score_summaries.update(base_net._proposal_targets)
    base_net._score_summaries.update(base_net._predictions)
    return self._merge_outputs

  # def get_stage_output(self):
  #   predictions = \
  #     dict((output_name, self._predictions[output_name][stage]) \
  #     for output_name in self._predictions)
  #   proposal_targets = \
  #     dict((output_name, self._proposal_targets[output_name][stage]) \
  #     for output_name in self._proposal_targets)


  # def _add_losses(self, sigma_rpn=3.0):
  #   for stage in self._stage_list:
  #     with tf.variable_scope('loss_' + stage) as scope:
  #     # RPN, class loss
  #     rpn_num = cfg.TRAIN.RPN_BATCHSIZE
  #     rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [rpn_num, 2])
  #     rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [rpn_num])
  #     rpn_select = tf.where(tf.not_equal(rpn_label, -1))
  #     rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [rpn_num, 2])
  #     rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [rpn_num])
  #     rpn_cross_entropy = tf.reduce_mean(
  #       tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
  #
  #     # RPN, bbox loss
