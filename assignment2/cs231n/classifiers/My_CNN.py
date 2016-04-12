# Import the packages we need:
import numpy as np

# Get layers, fast_layers, and layer_utils
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class FiveLayer_Convnet:
    """
  A three-layer convolutional network with the following architecture:
  
  [conv - spatial_batch - relu - 2x2 max pool - dropout]x4 - [affine - relu - dropout] - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 10, 5], filter_sizes=[7,5,3], 
               padding_sizes=[2,3,4], stride_sizes=[3,4,5],
               hidden_dims=100, num_classes=10, weight_scale=1e-3, reg=0.0,
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
    
    C,H,W = input_dims
    
    ################## Initializing W's: ##################
    self.params['W1'] = weight_scale * np.random.randn(num_filters[0], C, filter_sizes[0], filter_sizes[0])
    self.params['W2'] = weight_scale * np.random.randn(num_filters[1], num_filters[0], filter_sizes[1], filter_sizes[1])
    self.params['W3'] = weight_scale * np.random.randn(num_filters[2], num_filters[1], filter_sizes[2], filter_sizes[2])
        
    ''' Calculating size for next input(first affine) W '''
    H = filter_sizes[2]
    HH = filter_sizes[0]
    P = padding_sizes[0]
    S = stride_sizes[0]
    size_p = (H-2*P+HH)/S + 1
    
    # From the pooling layer
    size_p /= 2
    
    self.params['W4'] = weight_scale * np.random.randn(num_filters[2]*size_p*size_p, hidden_dim)
    self.params['W5'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    
    ################## Initializing b's: ##################
    self.params['b1'] = np.zeros((num_filters[0]))
    self.params['b2'] = np.zeros((num_filters[1]))
    self.params['b3'] = np.zeros((num_filters[2]))
    self.params['b4'] = np.zeros((hidden_dim))
    self.params['b4'] = np.zeros((num_classes))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
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
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out_forward_1, cache_forward_1 = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    out_forward_2, cache_forward_2 = affine_forward(out_forward_1, self.params['W2'], self.params['b2'])
    out_relu_2, cache_relu_2 = relu_forward(out_forward_2)
    scores, cache_forward_3 = affine_forward(out_relu_2, self.params['W3'], self.params['b3'])
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
    loss, dout = softmax_loss(scores, y)
    
    # Add regularization
    loss += self.reg * 0.5 * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2) + np.sum(self.params['W1']**2))

    dX3, grads['W3'], grads['b3'] = affine_backward(dout, cache_forward_3)
    dX2 = relu_backward(dX3, cache_relu_2)
    dX2, grads['W2'], grads['b2'] = affine_backward(dX2, cache_forward_2)
    dX1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dX2, cache_forward_1)

    grads['W3'] += self.reg * self.params['W3']
    grads['W2'] += self.reg * self.params['W2']
    grads['W1'] += self.reg * self.params['W1']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
        
    