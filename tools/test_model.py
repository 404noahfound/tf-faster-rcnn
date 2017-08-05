import tensorflow as tf

import _init_paths
from model.config import cfg
from model.test_model import test_model_graph
from nets.resnet_v1 import resnetv1

if __name__ == '__main__':
  net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=50)
  test_model_graph(net)
