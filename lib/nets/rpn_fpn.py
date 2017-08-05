import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.fpn import FeaturePyramidNetwork

class RPN_FPN(FeaturePyramidNetwork):

  def __init__(self, base_net, name='rpn_fpn'):
    FeaturePyramidNetwork.__init__(self, base_net, name)
    self._output_name_list = \
      ["rpn_cls_score", 'rpn_cls_score_reshape', 'rpn_cls_prob',
        'rpn_cls_pred', 'rpn_bbox_pred', 'rpn_bbox_pred', 'rois']
    self._net_map = {
                  #  'C1':'resnet_v1_50/conv1/Relu:0',
                   'C2':'resnet_v1_50/block1/unit_2/bottleneck_v1',
                   'C3':'resnet_v1_50/block2/unit_3/bottleneck_v1',
                   'C4':'resnet_v1_50/block3/unit_5/bottleneck_v1',
                   'C5':'resnet_v1_50/block4/unit_3/bottleneck_v1',
      }
    self._net_begin = 2

  def build_rpn_head(self, base_layer):
    base_net = self._base_net
    initializer, _ = self.set_initializers()
    is_training = self._is_training
    net_conv = base_layer
    # TODO:implement anchor_component
    num_anchors = base_net._anchor_component()
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
      rpn_labels = base_net._anchor_target_layer(rpn_cls_score, "anchor")
      # Try to have a deterministic order for the computing graph,
      # for reproducibility
      with tf.control_dependencies([rpn_labels]):
        rois, _ = base_net._proposal_target_layer(rois, roi_scores, "rpn_rois")

    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = base_net._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError
    output = {}
    output["rpn_cls_score"] = rpn_cls_score_raw
    output["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    output["rpn_cls_prob"] = rpn_cls_prob_reshape
    output["rpn_cls_pred"] = rpn_cls_pred
    output["rpn_bbox_pred"] = rpn_bbox_pred
    output["rois"] = rois
    return rois, output


  def build_heads(self):
    super(RPN_FPN, self).build_heads(build_rpn_head, 'RPN_FPN')
