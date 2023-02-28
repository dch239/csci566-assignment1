from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=3,
                stride=1, padding=0, init_scale=.02, name="conv"),
            MaxPoolingLayer(pool_size=2, stride=2, name="maxpool"),
            flatten(name="flat"),
            fc(input_dim=27, output_dim=5, init_scale=0.02, name="fc")
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=5, number_filters=16, stride=1, padding=1, init_scale=0.02,
                        name="conv1"),
            gelu(name="gelu1"),
            MaxPoolingLayer(pool_size=2, stride=2, name="pool1"),

            ConvLayer2D(input_channels=16, kernel_size=3, number_filters=32, stride=2, padding=1, init_scale=0.02,
                        name="conv2"),
            gelu(name="gelu2"),
            MaxPoolingLayer(pool_size=2, stride=2, name="pool2"),
            # dropout(name="dropout1", keep_prob=0.25, seed=1234),
            #
            # ConvLayer2D(input_channels=64, kernel_size=3, number_filters=32, stride=1, padding=0, init_scale=0.02,
            #             name="conv1"),
            # gelu(name="gelu1"),
            # ConvLayer2D(input_channels=32, kernel_size=3, number_filters=64, stride=1, padding=0, init_scale=0.02,
            #             name="conv2"),
            # gelu(name="gelu2"),
            # MaxPoolingLayer(pool_size=2, stride=2, name="pool1"),
            # dropout(name="dropout1", keep_prob=0.25, seed=1234),



            flatten(name="flatten1"),
            dropout(keep_prob=0.75, seed=seed),
            # fc(1568, 512, init_scale=0.02, name="fc4"),
            # gelu(name="gelu4"),
            # dropout(name="dropout4", keep_prob=0.5, seed=1234),
            fc(512, 128, init_scale=0.02, name="fc1"),
            # gelu(name="gelu3"),
            # dropout(name="dropout2", keep_prob=0.5, seed=1234),
            # fc(512, 128, init_scale=0.02, name="fc2"),
            # gelu(name="gelu2"),
            # dropout(name="dropout2", keep_prob=keep_prob, seed=seed),
            fc(128, 20, init_scale=0.02, name="fc3")
            ########### END ###########
        )