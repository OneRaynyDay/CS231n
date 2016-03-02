import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  
    Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # let's first take care of the numerical instability:
  # we have to find the maximum of f_yi, where f = X.dot(W).
  # in other words, we need to find the maximum of the X.dot(W) score for every sample
  for i in range(N):
      cur_X = X[i,:] # we select X_i, which is a 1xD matrix
      f_scores = cur_X.dot(W)
      f_max = np.max(f_scores)
      # now we want to subtract everything by the f_max so we get a smaller f_scores with the same proportion
      f_scores -= f_max
      
      # Second, let's take care of the cost function, where it's equal to log(e^(score_i)/sum(e^(score_all)))
      loss += -f_scores[y[i]] + np.log(np.sum(np.exp(f_scores)))
      
      # Thirdly, let's take care of the cost function prime for W_i
      # it's equal to X_i(-1 + e^(score_i)/(sum(e^(score_all))))
      # important: I was confused about this but after thinking about it:
      # The COLUMN of W are the classes. You assign these d/dW_j for j = class # 
      # and to compute them you get a [1 x D] matrix from 
      # X_i * (scalar of np.exp(f_scores[j])/(np.sum(np.exp(f_scores)))
      for j in range(C):
        dW[:,j] += cur_X*(-1*(j==y[i]) + np.exp(f_scores[j])/(np.sum(np.exp(f_scores))))
      
  loss /= N # we need to average out the sample
  dW /= N
  loss += reg*np.sum(W**2)/2  # we need to add the regularization terms
  dW += reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
   
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Step 1: let's remove the numeric instability
  f = X.dot(W)
  # remove the max of each score column
  f_max = np.max(f).reshape(-1, 1)
  f -= f_max
  scores = np.exp(f)
    
  # Step 2: let's compute the loss
  # summing everything across the # of samples
  scores_sums = np.sum(scores, axis=1)
  # select all the valid scores
  scores_correct = scores[np.arange(N), y]
  f_correct = f[np.arange(N), y]
  loss = np.sum(-f_correct+np.log(scores_sums))

  # Step 3: let's compute the gradient of the function
  # We need to first take the scores of all cells - already done by scores
  # afterwards, we need to divide all of them row-wise by the sum of each row's scores
  sum = scores/(scores_sums.reshape(-1,1))
  # later on, we're gonna need a binary matrix for adding the 1's inside of the dW[:,j]
  bi_matrix = np.zeros_like(scores)
  bi_matrix[np.arange(N), y] = -1
  
  # Then, recall we need to either add 1 or subtract 1 to each element if it's in the correct class
  sum += bi_matrix

  # Then, we will multiply it elementwise by X_i(this is kind of weird) to get a 3D array of NxDxC
  dW = (X.T).dot(sum)

  # Don't forget the regularization
  loss /= N
  loss += reg*np.sum(W**2)/2
  dW /= N
  dW += reg*W
   
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

