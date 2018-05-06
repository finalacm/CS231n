from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.params['W1'] = np.random.randn(
            *(num_filters, C, filter_size, filter_size)) * weight_scale
        self.params['b1'] = np.zeros((num_filters, ))
        
        H = H // 2
        W = W // 2
        
        self.params['W2'] = np.random.randn(
            *(num_filters * H * W, hidden_dim)) * weight_scale
        self.params['b2'] = np.zeros((hidden_dim, ))
        
        self.params['W3'] = np.random.randn(
            *(hidden_dim, num_classes)) * weight_scale
        self.params['b3'] = np.zeros((num_classes, ))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        N, C, H, W = X.shape
        
        conv_relu_pool_output, conv_relu_pool_cache = conv_relu_pool_forward(
            X, W1, b1, conv_param, pool_param)
        
        # conv_output, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
        
        # first_relu_output, first_relu_cache = relu_forward(conv_output)
        
        # pool_output, pool_cache = max_pool_forward_naive(
            # first_relu_output, pool_param)
        
        # hidd_output, hidd_cache = affine_forward(pool_output, W2, b2)
        
        hidd_output, hidd_cache = affine_forward(conv_relu_pool_output, W2, b2)
        
        second_relu_output, second_relu_cache = relu_forward(hidd_output)
        
        scores, lastcache = affine_forward(second_relu_output, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        reg = self.reg
        loss, ds = softmax_loss(scores, y)
        
        loss += 0.5 * reg * np.sum(W3 * W3)
        
        ds, dw, db = affine_backward(ds, lastcache)
        dw += reg * np.sum(W3)
        grads['W3'], grads['b3'] = dw, db
        
        ds = relu_backward(ds, second_relu_cache)

        loss += 0.5 * reg * np.sum(W2 * W2)
        ds, dw, db = affine_backward(ds, hidd_cache)
        dw += reg * np.sum(W2)
        grads['W2'], grads['b2'] = dw, db

        # ds = max_pool_backward_naive(ds, pool_cache)
        
        # ds = relu_backward(ds, first_relu_cache)
        
        loss += 0.5 * reg * np.sum(W1 * W1)
        # ds, dw, db = conv_backward_naive(ds, conv_cache)
        ds, dw, db = conv_relu_pool_backward(ds, conv_relu_pool_cache)
        dw += reg * np.sum(W1)
        grads['W1'], grads['b1'] = dw, db
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
