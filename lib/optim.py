from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)
        
        #### MLP ####
        if not hasattr(self.net, "preprocess") and \
           not hasattr(self.net, "rnn") and \
           not hasattr(self.net, "postprocess"):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, weight_decay=0.0):
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layer):
        for n, dv in layer.grads.items():
            #############################################################################
            # TODO: Implement the SGD with (optional) Weight Decay                      #
            #############################################################################
            # dv += self.weight_decay * layer.params[n]
            # layer.params[n] -= self.lr * dv
            w = layer.params[n]

            # Compute the weight decay term (optional)
            decay = self.weight_decay * w

            # Compute the update term based on the gradient and learning rate
            update = self.lr * dv

            # Update the weight by subtracting the update term and adding the decay term
            w -= update + decay

            # Save the updated weight back into the layer
            layer.params[n] = w
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################



class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8, weight_decay=0.0):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t
        self.weight_decay=weight_decay

    def update(self, layer):
        #############################################################################
        # TODO: Implement the Adam with [optinal] Weight Decay                      #
        #############################################################################
        for param_name, param in layer.params.items():
            if param_name not in self.mt:
                self.mt[param_name] = np.zeros_like(param.data)
                self.vt[param_name] = np.zeros_like(param.data)

            grad = layer.grads[param_name]

            if self.weight_decay > 0:
                w = layer.params[param_name]
                grad += self.weight_decay * w

            self.t += 1
            lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

            self.mt[param_name] = self.beta1 * self.mt[param_name] + (1 - self.beta1) * grad
            self.vt[param_name] = self.beta2 * self.vt[param_name] + (1 - self.beta2) * (grad ** 2)

            delta_param = - lr_t * self.mt[param_name] / (np.sqrt(self.vt[param_name]) + self.eps)
            layer.params[param_name].data += delta_param
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
