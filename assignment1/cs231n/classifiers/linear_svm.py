import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  print(W.shape)
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #print(dW[:,j].shape)
        #print(X[i,:].shape)
        #print(dW[:,j] - X[i,:])
        dW[:,y[i]] -= X[i,:]
        dW[:,j] += X[i,:]

    
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_samples = X.shape[0]
  num_features = X.shape[1]
  num_classes = y.shape[0]
  # print("W's shape : ")
  # print(W.shape)
  # print("X's shape : ")
  # print(X.shape)
  # print("y's shape : ")
  # print(y.shape)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # first, we want to get the scores of every single sample.
  scores = X.dot(W)
  # print("Scores : ")
  # print(scores.shape)
  # next, pull out the correct answers
  correct_class_scores = scores[np.arange(num_samples) , y].reshape(num_samples,-1)
  # print("Correct_class_scores : ")
  # print(correct_class_scores.shape)
  # next, we need to subtract the scores of every single sample by the correct classes (we also add delta)
  # this also means that the correct class's score columns would be 0.
  new_scores = np.maximum(scores - correct_class_scores + 1.0, 0) # we use 1.0 for delta
  new_scores[np.arange(num_samples) , y] = 0
  # print("Our new scores's size : " + str(new_scores.shape))
  # ====== now that we have the scores, we can find the cost! ======
  
  # we don't want to add the correct score's score. We want 
  # don't forget regularization terms
  loss = np.sum(new_scores)/num_samples + 0.5 * reg * np.sum(W * W)  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
    
  # make a binary matrix
  binary_scores = np.array(new_scores != 0, dtype=np.float32)
  # print("Our binary score's size : " + str(binary_scores.shape))
  
  # sum them column wise. This gets us the # of times we want to change for dWj
  binary_score_col = np.sum(binary_scores, axis = 1)
  # print("By summing up our binary score, we get : " + str(binary_score_col))
  # print("Binary_score_col size: " + str(binary_score_col.shape))
  # print("xrange(num_samples) : " + str(num_samples))
  # print("y : " + str(y.shape))
  binary_scores[np.arange(num_samples), y] = -binary_score_col
  
  dW = X.T.dot(binary_scores)/num_samples + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
